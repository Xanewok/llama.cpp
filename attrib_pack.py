#!/usr/bin/env python3
"""Attribution pack parser: nsys sqlite exports -> per-axis time budget with physical floors and headroom.

Axes (ms per speculative round, per GPU, partitioned so the axes sum to the round):
  FWD   weight-streaming kernels (GEMV/GEMM incl. lm_head)        floor = streamed weight bytes / (tp * BW)
  KV    attention kernels + GDN/conv state kernels                floor = KV bytes at depth + GDN state bytes / (tp * BW)
  COMM  AllReduce kernels                                          floor = calls * measured AR latency at the payload size
  ELEM  norm/activation/quantize/elementwise kernels               floor = demonstrated (other engine), not physical
  STEP  idle gaps (host), sampling kernels, memcpy                 floor = demonstrated (other engine), not physical
Draft (MTP phases) and target (V phase incl. its trailing gap) are separate columns. A kernel interval is charged to one
axis only (priority FWD > KV > COMM > ELEM > SAMP > COPY; no kernel running = idle), so axes sum to the window exactly.

Contract: a trace that lacks the CUPTI kernel/runtime tables, lacks a device, or yields fewer than --min-rounds complete
steady rounds is rejected (Reject), never summarised.
"""
import argparse
import bisect
import collections
import hashlib
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCH = {'cudaGraphLaunch', 'cudaLaunchKernel', 'cudaLaunchKernelExC', 'cuLaunchKernel', 'cuLaunchKernelEx'}
SYNC = {'cudaStreamSynchronize', 'cudaEventSynchronize'}
PRIORITY = ('FWD', 'KV_ATTN', 'KV_GDN', 'COMM', 'ELEM', 'SAMP', 'COPY')
CLASS_ORDER = PRIORITY + ('IDLE',)
AXIS_OF = {'FWD': 'FWD', 'KV_ATTN': 'KV', 'KV_GDN': 'KV', 'COMM': 'COMM', 'ELEM': 'ELEM',
           'SAMP': 'STEP', 'COPY': 'STEP', 'IDLE': 'STEP'}
AXES = ('FWD', 'KV', 'COMM', 'ELEM', 'STEP')
# AR latency per call (us) by payload KiB: best demonstrated, not physical. 10 KiB 9.0 (host-staged 9.3, graph-launched NCCL LL
# 9.2 at 8 KiB); 40 KiB 14.0 (llama.cpp V phase, round-trace 10-01: 14.4 us mean per call; item-47 build 17.0, NCCL RING_LL 27,
# host-staged 17.5). Beyond 40 KiB the line is extrapolated.
AR_US = ((10.0, 9.0), (40.0, 14.0))


class Reject(Exception):
    """The trace cannot support an honest budget."""


# ---------------------------------------------------------------------------------------------------------- loading
def open_ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)


def classify(name):
    n = name.lower()
    # ggml_cuda_ar_add_kernel / the bf16 convert are the internal AllReduce's reduce and cast stages.
    if 'nccl' in n or 'ggml_cuda_ar_' in n or 'hsar' in n or 'convert_unary_cont_vec4<float, __nv_bfloat16>' in n:
        return 'COMM'
    if re.search(r'topk|top_k|top_p|topp|min_p|gumbel|rejection|resample|logits_stats|softmax|radix|maxpairs|sampled_and_rejected|'
                 r'argsort|_ranks_kernel|recovered|(?<![a-z0-9])_(?:candidates|select)_kernel', n):
        return 'SAMP'
    # concat_non_cont / k_get_rows_float_vec / float->float cpy_scalar: llama.cpp's GDN conv/ssm state gather, concat and
    # write-back (48 calls per forward each for the first two). cpy_scalar float->float also serves ggml_cont of
    # activations, so KV_GDN is an upper bound for it (inferred: <= 0.5 ms/round on the item-47 trace).
    # The HQ wmma bf16 16x16_128x1 GEMM (48 calls/step) is the GDN in_proj_a/b weight product: FWD, not KV.
    if re.search(r'gated_delta|delta_rule|causal_conv|ssm_conv|ssm_|mamba|chunk_gated|chunk_fwd|chunk_local|chunk_scaled|'
                 r'solve_tril|recompute_w_u|wy_fast|fwd_kernel_h|fla_|concat_non_cont|k_get_rows_float_vec|'
                 r'cpy_scalar<&cpy_1_scalar<float, float>>', n):
        return 'KV_GDN'
    if re.search(r'flash_attn|kernel_mha|fattn|flashinfer|batchprefill|batchdecode|reshape_and_cache|qk_rmsnorm_rope|set_rows|'
                 r'flash_fwd|rotary|rope_|dequantize_block_q8_0_f16|fwht_cuda', n):
        return 'KV_ATTN'
    if re.search(r'marlin::|mul_mat_q|mul_mat_vec|cutlass|gemv|gemm|cublas|xmma|splitk|wvsplit|machete', n):
        # Inductor epilogues carry "marlin_gemm" in their names but only do element-wise work.
        return 'ELEM' if n.startswith('triton_') else 'FWD'
    return 'ELEM'


def shortname(n):
    """Compact llama.cpp kernel names for tables (mul_mat_vec_q<type, ncols> keeps its template shape)."""
    m = re.search(r'mul_mat_vec_q<\(ggml_type\)(\d+), \(int\)(\d+)', n)
    if m:
        return f'mmvq{m.group(1)}x{m.group(2)}'
    m = re.search(r'mul_mat_q<\(ggml_type\)(\d+)', n)
    if m:
        return f'mmq{m.group(1)}'
    if 'gated_delta_net' in n:
        return 'gdn'
    m = re.match(r'(?:void )?([A-Za-z_0-9:]+)', n)
    return m.group(1) if m else n[:40]


class Trace:
    """Kernels as tuples (start, end, dev, nameId, cid, pid, graphId); per-device sorted copies for window queries."""

    def __init__(self, path, need_memcpy=False, need_runtime=True):
        try:
            self._load(path, need_memcpy, need_runtime)
        except sqlite3.DatabaseError as e:
            raise Reject(f'not a readable sqlite export (truncated or corrupt?): {e}')

    def _load(self, path, need_memcpy, need_runtime):
        self.path = Path(path).resolve()
        try:
            db = open_ro(path)
            tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        except sqlite3.DatabaseError as e:
            raise Reject(f'not a readable sqlite export: {e}')
        need = ['CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'] + (['CUPTI_ACTIVITY_KIND_RUNTIME'] if need_runtime else []) + (
            ['CUPTI_ACTIVITY_KIND_MEMCPY'] if need_memcpy else [])
        miss = [t for t in need if t not in tables]
        if miss:
            raise Reject('missing tables: ' + ', '.join(miss))
        self.names = dict(db.execute('select id, value from StringIds'))
        cols = {r[1] for r in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
        gcol = 'graphId' if 'graphId' in cols else 'NULL'
        pcol = 'globalPid' if 'globalPid' in cols else '0'
        self.kernels = db.execute(f'select start,end,deviceId,demangledName,correlationId,{pcol},{gcol} '
                                  'from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
        if not self.kernels:
            raise Reject('kernel table is empty')
        self.runtime = [(s, e, re.sub(r'_v\d+$', '', self.names.get(n, '')), cid, tid)
                        for s, e, n, cid, tid in db.execute('select start,end,nameId,correlationId,globalTid '
                                                            'from CUPTI_ACTIVITY_KIND_RUNTIME order by start')] if need_runtime else []
        self.memcpy, self.memset = [], []
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            self.memcpy = db.execute('select start,end,deviceId,bytes,copyKind,correlationId,globalPid '
                                     'from CUPTI_ACTIVITY_KIND_MEMCPY order by start').fetchall()
        if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
            self.memset = db.execute('select start,end,deviceId from CUPTI_ACTIVITY_KIND_MEMSET order by start').fetchall()
        db.close()
        self.cls = {nid: classify(self.names[nid]) for nid in {k[3] for k in self.kernels}}
        self.devs = sorted({k[2] for k in self.kernels})
        self.t0 = self.kernels[0][0]
        self.t1 = max(k[1] for k in self.kernels)
        self.by_dev = {d: [k for k in self.kernels if k[2] == d] for d in self.devs}
        self.starts = {d: [k[0] for k in v] for d, v in self.by_dev.items()}
        self.maxdur = {d: max(k[1] - k[0] for k in v) for d, v in self.by_dev.items()}
        self.copies = {d: [(m[0], m[1]) for m in self.memcpy if m[2] == d] + [(m[0], m[1]) for m in self.memset if m[2] == d]
                       for d in self.devs}
        self.cstarts = {d: sorted(c[0] for c in v) for d, v in self.copies.items()}
        for d in self.devs:
            self.copies[d].sort()

    def sha(self):
        h = hashlib.sha256()
        with open(self.path, 'rb') as f:
            h.update(f.read(1 << 20))
        return h.hexdigest()[:16] + f':{self.path.stat().st_size}'

    def window_kernels(self, d, a, b):
        i0 = bisect.bisect_left(self.starts[d], a - self.maxdur[d])
        out = []
        for k in self.by_dev[d][i0:]:
            if k[0] >= b:
                break
            if k[1] > a:
                out.append(k)
        return out

    def window_copies(self, d, a, b):
        cs = self.copies[d]
        i0 = max(0, bisect.bisect_left(self.cstarts[d], a) - 64)
        return [c for c in cs[i0:] if c[0] < b and c[1] > a]


def partition(tr, d, a, b, priority=PRIORITY):
    """ns of [a,b) per class on device d; exact cover of the window. Returns (ns by class, kernel-sum ns by class)."""
    if b <= a:
        raise ValueError('empty window')
    ev = []
    ksum = collections.Counter()
    ks = tr.window_kernels(d, a, b)
    for k in ks:
        c = tr.cls[k[3]]
        s, e = max(a, k[0]), min(b, k[1])
        ev.append((s, 1, c))
        ev.append((e, -1, c))
        ksum[c] += e - s
    for s0, e0 in tr.window_copies(d, a, b):
        s, e = max(a, s0), min(b, e0)
        if e > s:
            ev.append((s, 1, 'COPY'))
            ev.append((e, -1, 'COPY'))
    ev.sort(key=lambda x: (x[0], x[1]))
    cnt = collections.Counter()
    out = collections.Counter()
    t = a
    i = 0
    n = len(ev)
    while i < n:
        te = ev[i][0]
        if te > t:
            act = next((c for c in priority if cnt[c] > 0), 'IDLE')
            out[act] += te - t
            t = te
        while i < n and ev[i][0] == te:
            cnt[ev[i][2]] += ev[i][1]
            i += 1
    if b > t:
        out['IDLE'] += b - t
    if sum(out.values()) != b - a:
        raise ValueError('accounting does not cover the window')
    return out, ksum


# ---------------------------------------------------------------------------------------------------- engine detection
def detect_engine(tr):
    sample = ' '.join(tr.names[n] for n in list(tr.cls)[:400]).lower()
    if 'ggml' in sample or 'mul_mat_vec_q' in sample or 'gated_delta_net' in sample:
        return 'llama'
    if 'marlin' in sample or 'triton_' in sample or 'vllm' in sample:
        return 'hq'
    raise Reject('engine not recognised from kernel names (pass --engine)')


# ----------------------------------------------------------------------------------------------------- round finding
def llama_blocks(tr, head_type):
    """Host launch blocks: launch runs that end at the next non-peer H2D input copy (attrib_v2 definition)."""
    by_cid = collections.defaultdict(list)
    for k in tr.kernels:
        by_cid[(k[5], k[4])].append(k)
    ck = {(m[6], m[5]): m[4] for m in tr.memcpy}
    blocks, cur = [], None
    for s, e, nm, cid, tid in tr.runtime:
        key = ((tid >> 24) << 24, cid)
        pid_key = key
        if nm in LAUNCH:
            if cur is None:
                cur = {'host_s': s, 'host_e': e, 'kernels': [], 'calls': []}
            cur['host_e'] = e
            cur['calls'].append((s, e, nm))
            cur['kernels'].extend(by_cid.get(pid_key, []))
        elif 'Memcpy' in nm and 'Peer' not in nm and cur is not None and ck.get(pid_key) == 1:
            cur['h2d'] = (s, e)
            blocks.append(cur)
            cur = None
    if cur is not None:
        blocks.append(cur)
    head = f'mul_mat_vec_q<(ggml_type){head_type}, (int)1,' if head_type else None
    for b in blocks:
        ns = [tr.names[k[3]] for k in b['kernels']]
        b['kind'] = ('V' if any('gated_delta_net' in n for n in ns) else
                     'D' if head and any(head in n for n in ns) else
                     'C' if any('flash_attn' in n for n in ns) else '?')
        b['dev'] = {d: [k for k in b['kernels'] if k[2] == d] for d in tr.devs}
    return blocks


def llama_rounds(tr, head_type, k_force=None, trim=0.03, skip_ms=750.0):
    blocks = llama_blocks(tr, head_type)
    if not blocks:
        raise Reject('no launch blocks found')
    vi = [i for i, b in enumerate(blocks) if b['kind'] == 'V']
    shapes = collections.Counter()
    cand = []
    for a, b in zip(vi, vi[1:]):
        sel = blocks[a:b]
        shape = ''.join(x['kind'] for x in sel)
        shapes[shape] += 1
        cand.append((shape, a, b))
    if not cand:
        raise Reject('fewer than two verify (V) blocks')
    def ok_shape(sh):
        m = re.fullmatch(r'VC?(D+)', sh)
        if not m:
            return False
        return k_force is None or len(m.group(1)) == k_force
    good = [c for c in cand if ok_shape(c[0])]
    if not good:
        raise Reject(f'no round matches V C? D^k (k={k_force}); block shapes seen: {dict(shapes.most_common(4))}')
    modal = collections.Counter(c[0] for c in good).most_common(1)[0][0]
    rounds = []
    for shape, a, b in good:
        if shape != modal:
            continue
        sel = blocks[a:b]
        names = []
        nd = 0
        for x in sel:
            if x['kind'] == 'D':
                nd += 1
                names.append(f'D{nd}')
            else:
                names.append(x['kind'])
        rounds.append({'host_s': sel[0]['host_s'], 'host_e': blocks[b]['host_s'], 'phases': list(zip(names, sel)),
                       'next': blocks[b]})
    span_ms = (tr.t1 - tr.t0) / 1e6
    cut = tr.t0 + (skip_ms * 1e6 if span_ms > 1000 else 0)
    after_cut = [r for r in rounds if r['host_s'] >= cut]
    steady = after_cut
    if trim and after_cut:
        med = statistics.median(r['host_e'] - r['host_s'] for r in after_cut)
        steady = [r for r in after_cut if abs(r['host_e'] - r['host_s'] - med) <= trim * med]
    info = {'block_shapes': dict(shapes.most_common(6)), 'modal_shape': modal, 'complete_rounds': len(rounds),
            'after_skip': len(after_cut), 'steady': len(steady), 'skip_ms': skip_ms if span_ms > 1000 else 0.0,
            'trim_frac': trim, 'k': modal.count('D'), 'has_C': 'C' in modal}
    return steady, blocks, info


def hq_rounds(tr, k_force=None, trim=0.03):
    gl = {((tid >> 24) << 24, cid): (s, e) for s, e, nm, cid, tid in tr.runtime if nm == 'cudaGraphLaunch'}
    groups = collections.defaultdict(list)
    for kk in tr.kernels:
        if kk[6] is not None:
            groups[(kk[2], kk[5], kk[4])].append(kk)
    if not groups:
        raise Reject('no graph-replayed kernels (HQ steps need --cuda-graph-trace=node)')
    launches = {d: [] for d in tr.devs}
    for (d, pid, cid), ks in groups.items():
        if (pid, cid) not in gl:
            raise Reject('graph kernel without a process-scoped cudaGraphLaunch call')
        launches[d].append({'s': min(x[0] for x in ks), 'e': max(x[1] for x in ks), 'n': len(ks), 'host_s': gl[(pid, cid)][0]})
    for d in launches:
        launches[d].sort(key=lambda x: x['s'])
    nmax = max(l['n'] for ls in launches.values() for l in ls)
    for ls in launches.values():
        for l in ls:
            l['kind'] = 'V' if l['n'] >= max(500, 0.5 * nmax) else 'D'
    vs = {d: [i for i, l in enumerate(launches[d]) if l['kind'] == 'V'] for d in tr.devs}
    if len({len(v) for v in vs.values()}) != 1:
        raise Reject('devices disagree on the number of verify graph replays')
    seqs = collections.Counter()
    out = []
    for idx in range(len(vs[tr.devs[0]]) - 1):
        per = {}
        shapes = set()
        for d in tr.devs:
            a, b = vs[d][idx:idx + 2]
            seq = launches[d][a:b + 1]
            shapes.add(''.join(x['kind'] for x in seq[:-1]))
            per[d] = seq
        if len(shapes) != 1:
            continue
        shape = shapes.pop()
        seqs[shape] += 1
        out.append((shape, per))
    good = [x for x in out if re.fullmatch(r'VD+', x[0]) and (k_force is None or len(x[0]) - 1 == k_force)]
    if not good:
        raise Reject(f'no step matches V D^k (k={k_force}); graph-launch shapes seen: {dict(seqs.most_common(4))}')
    modal = collections.Counter(x[0] for x in good).most_common(1)[0][0]
    steps = [x[1] for x in good if x[0] == modal]
    steps = steps[1:-1]  # capture boundaries
    if trim and steps:
        d0 = tr.devs[0]
        per = [s[d0][-1]['s'] - s[d0][0]['s'] for s in steps]
        med = statistics.median(per)
        steps = [s for s, p in zip(steps, per) if abs(p - med) <= trim * med]
    names = ['V'] + [f'D{i}' for i in range(1, len(modal))]
    rounds = []
    for per in steps:
        d0 = tr.devs[0]
        rounds.append({'host_s': per[d0][0]['host_s'], 'host_e': per[d0][-1]['host_s'],
                       'phases': [(n, {'dev_s': {d: per[d][i]['s'] for d in tr.devs}}) for i, n in enumerate(names)],
                       'dev_next': {d: per[d][-1]['s'] for d in tr.devs}})
    info = {'graph_shapes': dict(seqs.most_common(6)), 'modal_shape': modal, 'complete_rounds': len(out),
            'steady': len(rounds), 'trim_frac': trim, 'k': len(modal) - 1, 'has_C': False}
    return rounds, info


# ------------------------------------------------------------------------------------------------- decode attribution
def ar_floor_us(kib):
    (x0, y0), (x1, y1) = AR_US
    if kib <= x0:
        return y0
    return y0 + (y1 - y0) * (kib - x0) / (x1 - x0)


def phase_windows(tr, d, rnd, engine):
    """[(phase, a, b)] per device: from the phase's first kernel to the next phase's first kernel (trailing gap included)."""
    if engine == 'llama':
        starts = []
        for name, blk in rnd['phases']:
            ks = blk['dev'][d]
            if not ks:
                raise Reject(f'{name} block has no kernels on device {d}')
            starts.append(min(k[0] for k in ks))
        nxt = rnd['next']['dev'][d]
        if not nxt:
            raise Reject('next verify block has no kernels on device ' + str(d))
        starts.append(min(k[0] for k in nxt))
    else:
        starts = [ph[1]['dev_s'][d] for ph in rnd['phases']] + [rnd['dev_next'][d]]
    names = [p[0] for p in rnd['phases']]
    return [(n, starts[i], starts[i + 1]) for i, n in enumerate(names)]


def head_ns(tr, d, a, b):
    """Duration of the last weight kernel in [a,b): the output projection."""
    ks = [k for k in tr.window_kernels(d, a, b) if tr.cls[k[3]] == 'FWD' and k[0] >= a]
    return (ks[-1][1] - ks[-1][0]) if ks else 0


def gap_segments(tr, rnd, blocks_runtime):
    """llama: split each inter-block gap into readback / cpu_pre / inputs / launch latency (attrib_v2), device 0."""
    d = tr.devs[0]
    R = blocks_runtime
    starts = [r[0] for r in R['sync_runtime']]
    seg = collections.defaultdict(float)
    seq = [b for _, b in rnd['phases']] + [rnd['next']]
    h2d_calls = R['h2d']
    for b1, b2 in zip(seq, seq[1:]):
        k1, k2 = b1['dev'][d], b2['dev'][d]
        e1 = max(k[1] for k in k1)
        s2 = min(k[0] for k in k2)
        if s2 <= e1:
            continue
        tin = None
        i = bisect.bisect_left(R['h2d_starts'], b1['calls'][-1][1])
        if i < len(h2d_calls) and h2d_calls[i] < b2['host_s']:
            tin = h2d_calls[i]
        tin = tin if tin is not None else b2['host_s']
        sy = [x for x in R['syncs'] if x[1] >= e1 and x[0] < tin]
        trb = min(max(sy[0][1], e1), tin) if sy else e1
        marks = {'readback': (e1, trb), 'cpu_pre': (trb, tin), 'inputs': (tin, b2['host_s']), 'launch_lat': (b2['host_s'], s2)}
        for name, (x, y) in marks.items():
            x, y = max(x, e1), min(y, s2)
            if y > x:
                seg[name] += y - x
    return seg


def runtime_index(tr):
    R = {'syncs': [(s, e) for s, e, nm, cid, tid in tr.runtime if nm in SYNC]}
    ck = {(m[6], m[5]): m[4] for m in tr.memcpy}
    h2d = sorted(s for s, e, nm, cid, tid in tr.runtime
                 if 'Memcpy' in nm and 'Peer' not in nm and ck.get(((tid >> 24) << 24, cid)) == 1)
    R['h2d'] = h2d
    R['h2d_starts'] = h2d
    R['sync_runtime'] = []
    return R


def decode_budget(tr, model, depth, engine, k=None, min_rounds=30, trim=None, tok_per_round=None, draft_head_bytes=None,
                  skip_ms=750.0, unprofiled=None):
    if len(tr.devs) < model['tp']:
        raise Reject(f"trace has kernels on {len(tr.devs)} device(s), model expects tp={model['tp']}")
    trim = trim if trim is not None else (0.03 if engine == 'llama' else 0.05)
    if engine == 'llama':
        rounds, blocks, info = llama_rounds(tr, model.get('head_type'), k, trim, skip_ms)
        rt = runtime_index(tr)
    else:
        rounds, info = hq_rounds(tr, k, trim)
        rt = None
    if len(rounds) < min_rounds:
        raise Reject(f"only {len(rounds)} complete steady rounds (< {min_rounds}); {json.dumps(info)}")
    k = info['k']
    per_dev = {}
    for d in tr.devs[:model['tp']]:
        tot = collections.defaultdict(float)       # (phase_group, class) -> ns summed over rounds
        ksum = collections.defaultdict(float)
        counts = collections.defaultdict(float)    # n kernels / comm calls per phase
        durations = []
        head = collections.defaultdict(float)
        for r in rounds:
            wins = phase_windows(tr, d, r, engine)
            durations.append(wins[-1][2] - wins[0][1])
            for name, a, b in wins:
                grp = 'V' if name == 'V' else 'draft'
                part, ks = partition(tr, d, a, b)
                for c, v in part.items():
                    tot[(grp, c)] += v
                for c, v in ks.items():
                    ksum[(grp, c)] += v
                kk = tr.window_kernels(d, a, b)
                counts[('n_comm', name)] += sum(1 for x in kk if x[0] >= a and tr.cls[x[3]] == 'COMM')
                counts[('n_kern', name)] += sum(1 for x in kk if x[0] >= a)
                if name != 'C':
                    head[grp] += head_ns(tr, d, a, b)
        n = len(rounds)
        per_dev[d] = {'tot': {key: v / n / 1e6 for key, v in tot.items()},
                      'ksum': {key: v / n / 1e6 for key, v in ksum.items()},
                      'counts': {key: v / n for key, v in counts.items()},
                      'head': {g: v / n / 1e6 for g, v in head.items()},
                      'round_gpu_ms': statistics.fmean(durations) / 1e6,
                      'round_gpu_sd_ms': (statistics.stdev(durations) / 1e6 if n > 1 else 0.0)}
    d0 = tr.devs[0]
    host = [(r['host_e'] - r['host_s']) / 1e6 for r in rounds]
    res = per_dev[d0]
    phase_names = [p[0] for p in rounds[0]['phases']]
    n_draft = sum(1 for n in phase_names if n != 'V')
    n_head = sum(1 for n in phase_names if n.startswith('D'))  # the catch-up pass C has no output projection
    bw = model['bw_gbs'] * 1e9
    tp = model['tp']
    w = model['weights']
    dh = draft_head_bytes if draft_head_bytes is not None else w['draft_head_bytes']
    kv = model['kv']
    # ---- floors (ms per round, one GPU)
    floors = {}
    floors['FWD_V'] = w['target_stream_bytes'] / tp / bw * 1e3
    floors['FWD_draft'] = (n_draft * w['mtp_stream_bytes'] + n_head * dh) / tp / bw * 1e3
    floors['KV_V'] = (depth * kv['target_bytes_per_token'] + model['gdn']['state_bytes_per_forward'] * 2) / tp / bw * 1e3
    floors['KV_draft'] = n_draft * depth * kv['draft_bytes_per_token'] / tp / bw * 1e3
    hid = model['ar_bytes_per_token']
    ar = {'V': 0.0, 'draft': 0.0}
    ar_detail = {}
    for name, cnt in ((nm, c) for (kind, nm), c in res['counts'].items() if kind == 'n_comm'):
        toks = (k + 1) if name == 'V' else (2 if name == 'C' else 1)
        kib = toks * hid / 1024
        us = ar_floor_us(kib)
        ar['V' if name == 'V' else 'draft'] += cnt * us / 1e3
        ar_detail[name] = {'calls': cnt, 'payload_kib': kib, 'floor_us_per_call': us}
    floors['COMM_V'], floors['COMM_draft'] = ar['V'], ar['draft']
    floors['ELEM_V'] = model['elem_ref_ms_per_target_forward']
    floors['ELEM_draft'] = n_draft * model['elem_ref_ms_per_draft_forward']
    floors['STEP_V'] = model['step_ref_ms'] * 0.4
    floors['STEP_draft'] = model['step_ref_ms'] * 0.6
    rows = {}
    for ax in AXES:
        mv = sum(v for (g, c), v in res['tot'].items() if g == 'V' and AXIS_OF[c] == ax)
        md = sum(v for (g, c), v in res['tot'].items() if g == 'draft' and AXIS_OF[c] == ax)
        ks = sum(v for (g, c), v in res['ksum'].items() if AXIS_OF.get(c) == ax)
        fv, fd = floors[f'{ax}_V'], floors[f'{ax}_draft']
        rows[ax] = {'measured_ms': mv + md, 'target_ms': mv, 'draft_ms': md, 'kernel_sum_ms': ks,
                    'floor_ms': fv + fd, 'floor_target_ms': fv, 'floor_draft_ms': fd,
                    'headroom_ms': max(0.0, mv - fv) + max(0.0, md - fd)}
    round_ms = res['round_gpu_ms']
    for ax, r in rows.items():
        r['share_pct'] = 100 * r['measured_ms'] / round_ms
        r['headroom_pct_of_round'] = 100 * r['headroom_ms'] / round_ms
    sub = {}
    sub['FWD.lm_head'] = {'target_ms': res['head'].get('V', 0.0), 'draft_ms': res['head'].get('draft', 0.0)}
    for c in ('KV_ATTN', 'KV_GDN'):
        sub[c] = {'target_ms': res['tot'].get(('V', c), 0.0), 'draft_ms': res['tot'].get(('draft', c), 0.0)}
    for c in ('SAMP', 'COPY', 'IDLE'):
        sub['STEP.' + c] = {'target_ms': res['tot'].get(('V', c), 0.0), 'draft_ms': res['tot'].get(('draft', c), 0.0)}
    if engine == 'llama':
        seg = collections.defaultdict(float)
        for r in rounds:
            for kname, v in gap_segments(tr, r, rt).items():
                seg[kname] += v / len(rounds) / 1e6
        sub['STEP.gap_split'] = {k_: seg.get(k_, 0.0) for k_ in ('readback', 'cpu_pre', 'inputs', 'launch_lat')}
    skew = max(abs(per_dev[d]['round_gpu_ms'] - per_dev[d0]['round_gpu_ms']) for d in per_dev)
    floor_sum = sum(r['floor_ms'] for r in rows.values())
    out = {'kind': 'decode', 'engine': engine, 'depth': depth, 'k': k, 'n_rounds': len(rounds), 'tp': tp,
           'round_ms_gpu0': round_ms, 'round_sd_ms': res['round_gpu_sd_ms'], 'round_ms_host': statistics.fmean(host),
           'device_skew_ms': skew, 'axes': rows, 'sub': sub, 'floor_sum_ms': floor_sum,
           'ideal_speedup_x': round_ms / floor_sum if floor_sum else None,
           'ideal_speedup_basis': 'floors use the theoretical 448 GB/s per GPU, the GDN state read+write charge (guess) and, '
                                  'for ELEM and STEP, values demonstrated by HQ at k=3 (not physical limits)',
           'per_gpu': {str(d): {'round_ms': per_dev[d]['round_gpu_ms'],
                                **{ax: sum(v for (g, c), v in per_dev[d]['tot'].items() if AXIS_OF[c] == ax)
                                   for ax in ('FWD', 'KV', 'COMM')}} for d in per_dev},
           'ar': ar_detail, 'draft_forwards': n_draft, 'selection': info,
           'kernels_per_round': {n: c for (kind, n), c in res['counts'].items() if kind == 'n_kern'}}
    if tok_per_round:
        out['tok_per_round'] = tok_per_round
        out['ms_per_token'] = round_ms / tok_per_round
        out['tok_per_s'] = 1000 * tok_per_round / round_ms
    warn = []
    for ax in ('FWD', 'KV', 'COMM'):
        r = rows[ax]
        if r['measured_ms'] < 0.97 * r['floor_ms']:
            warn.append(f"{ax} measured {r['measured_ms']:.2f} ms is below its physical floor {r['floor_ms']:.2f} ms: "
                        'wrong model bytes / depth / bandwidth for this trace; headroom is clamped to 0')
    if res['round_gpu_sd_ms'] > 0.05 * round_ms:
        warn.append(f"round sd {res['round_gpu_sd_ms']:.2f} ms > 5% of the round: window not steady")
    if skew > 0.3:
        warn.append(f'GPU round times differ by {skew:.2f} ms')
    if len(rounds) < 40:
        warn.append(f'{len(rounds)} steady rounds (< 40): means are provisional')
    pg = out['per_gpu']
    for ax in ('FWD', 'COMM'):
        v = [x[ax] for x in pg.values()]
        if max(v) - min(v) > 0.3:
            warn.append(f'{ax} differs between GPUs by {max(v) - min(v):.2f} ms')
    if unprofiled:
        out['profiler_overhead'] = {'profiled_ms_per_round': round_ms, 'unprofiled_ms_per_round': unprofiled['ms'],
                                    'overhead_ms': round_ms - unprofiled['ms'],
                                    'overhead_pct': 100 * (round_ms - unprofiled['ms']) / unprofiled['ms'],
                                    'source': unprofiled['source']}
    out['warnings'] = warn
    out['per_device_round_ms'] = {str(d): per_dev[d]['round_gpu_ms'] for d in per_dev}
    out['top_kernels_gpu0'] = top_kernels(tr, d0, rounds, engine)
    return out


def top_kernels(tr, d, rounds, engine, n=14):
    agg = collections.Counter()
    cnt = collections.Counter()
    for r in rounds:
        a = min(w[1] for w in phase_windows(tr, d, r, engine))
        b = max(w[2] for w in phase_windows(tr, d, r, engine))
        for k in tr.window_kernels(d, a, b):
            if k[0] >= a:
                nm = shortname(tr.names[k[3]]) if engine == 'llama' else re.sub(r'\s+', ' ', tr.names[k[3]])[:60]
                agg[nm] += k[1] - k[0]
                cnt[nm] += 1
    nr = len(rounds)
    return [{'kernel': nm, 'ms_per_round': v / nr / 1e6, 'calls_per_round': cnt[nm] / nr} for nm, v in agg.most_common(n)]


# --------------------------------------------------------------------------------------------------- prefill budget
MARKERS = {'llama': re.compile(r'mul_mat_q<'),
           'hq': re.compile(r'chunk_gated_delta_rule_fwd|chunk_fwd_kernel_o|BatchPrefill|flash_fwd|fwd_kernel_h')}
PRE_CLASSES = ('GEMM', 'GDN', 'ATTN', 'COMM', 'ELEM', 'SAMP', 'COPY', 'IDLE')
PRE_MAP = {'FWD': 'GEMM', 'KV_GDN': 'GDN', 'KV_ATTN': 'ATTN', 'COMM': 'COMM', 'ELEM': 'ELEM', 'SAMP': 'SAMP', 'COPY': 'COPY',
           'IDLE': 'IDLE'}
PRE_PRIORITY = ('FWD', 'KV_GDN', 'KV_ATTN', 'COMM', 'ELEM', 'SAMP', 'COPY')


# Host work between ubatches (graph build, sampler, scheduler) leaves the GPU idle for 30-130 ms on llama (-ub 128, 2 GPUs): a
# threshold below that cuts one prefill into per-ubatch runs and the longest "run" is 1/150 of it (full-a1: 0.3 s of 45.7 s).
# Must stay above the per-step host gap and below any non-prefill activity that is allowed to split the window.
RUN_GAP_MS = 1000.0


def marker_runs(marked, gap_ns):
    runs, cur = [], [marked[0]]
    for k in marked[1:]:
        if k[0] - cur[-1][1] > gap_ns:
            runs.append(cur)
            cur = []
        cur.append(k)
    runs.append(cur)
    return runs


def prefill_budget(tr, model, engine, tokens, min_markers=64, window_ms=None, batch_tokens=None, run_gap_ms=RUN_GAP_MS):
    """Window = the longest span of prefill marker kernels, or --window. Shares by class, overall and per quarter of the prompt."""
    d = tr.devs[0]
    mk = MARKERS[engine]
    marked = [k for k in tr.by_dev[d] if mk.search(tr.names[k[3]])]
    runs, warn = [], []
    if window_ms:
        a, b = tr.t0 + window_ms[0] * 1e6, tr.t0 + window_ms[1] * 1e6
    else:
        if len(marked) < min_markers:
            raise Reject(f'only {len(marked)} prefill marker kernels (< {min_markers}); not a prefill trace')
        runs = marker_runs(marked, run_gap_ms * 1e6)
        best = max(runs, key=lambda r: r[-1][1] - r[0][0])
        if len(best) < min_markers:
            raise Reject(f'longest contiguous prefill run has {len(best)} marker kernels (< {min_markers})')
        a, b = best[0][0], best[-1][1]
        outside = sum(1 for k in marked if k[1] <= a or k[0] >= b)
        if outside > 0.05 * len(marked):
            warn.append(f'{outside} of {len(marked)} prefill marker kernels lie outside the chosen window '
                        f'({len(runs)} runs split at gaps > {run_gap_ms:g} ms): window may be a fragment of the prefill')
    if len(tr.devs) < model['tp']:
        raise Reject(f"trace has kernels on {len(tr.devs)} device(s), model expects tp={model['tp']}")
    part, ks = partition(tr, d, a, b, PRE_PRIORITY)
    # quarters of progress: equal counts of weight-GEMM launches (each batch issues the same GEMM sequence)
    gem = [k for k in tr.window_kernels(d, a, b) if tr.cls[k[3]] == 'FWD' and k[0] >= a]
    if len(gem) < 8:
        raise Reject('too few weight-GEMM kernels in the prefill window')
    cuts = [a] + [gem[len(gem) * q // 4][0] for q in (1, 2, 3)] + [b]
    quarters = [partition(tr, d, cuts[q], cuts[q + 1], PRE_PRIORITY)[0] for q in range(4)]
    total = b - a

    def shares(p):
        t = sum(p.values())
        return {PRE_MAP[c]: 100 * v / t for c, v in p.items()}

    def ms(p):
        return {PRE_MAP[c]: v / 1e6 for c, v in p.items()}
    pre = model['prefill']
    bt = batch_tokens or pre['batch_tokens']
    tp = model['tp']
    bw = model['bw_gbs'] * 1e9
    n_batches = tokens / bt
    restream_ms = n_batches * model['weights']['target_stream_bytes'] / tp / bw * 1e3
    lin = pre['params_flops_per_token'] * tokens
    attn = 2 * pre['attn_layers'] * pre['q_heads'] * pre['head_dim'] * tokens * tokens
    compute_ms = (lin + attn) / tp / (pre['peak_tops'] * 1e12) * 1e3
    wall_ms = total / 1e6
    mid_depths = [int(tokens * (q + .5) / 4) for q in range(4)]
    out = {'kind': 'prefill', 'engine': engine, 'tokens': tokens, 'window_ms': wall_ms, 'tok_per_s': tokens / wall_ms * 1e3,
           'ms_per_1k_tokens': wall_ms / tokens * 1e3, 'batch_tokens': bt, 'n_markers': len(marked), 'n_runs': len(runs),
           'warnings': warn, 'ms_by_class': ms(part), 'share_pct': shares(part),
           'kernel_sum_share_pct': {PRE_MAP[c]: 100 * v / sum(ks.values()) for c, v in ks.items()},
           'quarters': [{'approx_depth_tokens': mid_depths[q], 'share_pct': shares(quarters[q]),
                         'window_ms': sum(quarters[q].values()) / 1e6} for q in range(4)],
           'floors_ms': {'weight_restream': restream_ms, 'compute': compute_ms, 'ideal': max(restream_ms, compute_ms)},
           'floor_notes': {'weight_restream': f'{n_batches:.0f} batches of {bt} tokens, each re-reads target weights / (tp*BW)',
                           'compute': f"2*params*T + causal attention FLOPs over tp * {pre['peak_tops']} TOPS ({pre['peak_note']})"},
           'headroom_ms': max(0.0, wall_ms - max(restream_ms, compute_ms)),
           'headroom_pct_of_window': 100 * max(0.0, wall_ms - max(restream_ms, compute_ms)) / wall_ms}
    top = collections.Counter()
    cn = collections.Counter()
    for k in tr.window_kernels(d, a, b):
        nm = shortname(tr.names[k[3]]) if engine == 'llama' else re.sub(r'\s+', ' ', tr.names[k[3]])[:60]
        top[nm] += k[1] - k[0]
        cn[nm] += 1
    out['top_kernels_gpu0'] = [{'kernel': nm, 'ms': v / 1e6, 'calls': cn[nm]} for nm, v in top.most_common(14)]
    return out


# ------------------------------------------------------------------------------------------------------- reporting
def fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def decode_md(t, title):
    L = [f'### {title}', '']
    L.append(f"engine {t['engine']}, depth {t['depth']}, k={t['k']}, {t['n_rounds']} steady rounds, round {t['round_ms_gpu0']:.2f} ms "
             f"(GPU0 window, sd {t['round_sd_ms']:.2f}; host launch-to-launch {t['round_ms_host']:.2f}; GPU1 differs by <= "
             f"{t['device_skew_ms']:.2f} ms), draft forwards per round {t['draft_forwards']}"
             + (f", {t['tok_per_s']:.1f} tok/s at {t['tok_per_round']:.2f} tok/round" if 'tok_per_round' in t else ''))
    L += ['', '| axis | ms/round | % round | target ms | draft ms | kernel-sum ms | floor ms | headroom ms | headroom % round |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ax in AXES:
        r = t['axes'][ax]
        L.append(f"| {ax} | {r['measured_ms']:.2f} | {r['share_pct']:.1f} | {r['target_ms']:.2f} | {r['draft_ms']:.2f} | "
                 f"{r['kernel_sum_ms']:.2f} | {r['floor_ms']:.2f} | {r['headroom_ms']:.2f} | {r['headroom_pct_of_round']:.1f} |")
    tot = sum(r['measured_ms'] for r in t['axes'].values())
    L.append(f"| **sum** | {tot:.2f} | 100.0 | {sum(r['target_ms'] for r in t['axes'].values()):.2f} | "
             f"{sum(r['draft_ms'] for r in t['axes'].values()):.2f} | | {t['floor_sum_ms']:.2f} | "
             f"{sum(r['headroom_ms'] for r in t['axes'].values()):.2f} | |")
    L += ['', f"All floors sum to {t['floor_sum_ms']:.2f} ms: ideal speedup {fmt(t['ideal_speedup_x'])}x if every axis hit its floor "
          f"({t['ideal_speedup_basis']}; FWD and KV are additive memory traffic).", '']
    L.append('Per GPU (ms/round): ' + '; '.join(
        f"GPU{d}: FWD {v['FWD']:.2f}, COMM {v['COMM']:.2f}, KV {v['KV']:.2f}, round {v['round_ms']:.2f}" for d, v in t['per_gpu'].items()))
    if 'profiler_overhead' in t:
        o = t['profiler_overhead']
        L.append(f"Profiler overhead (profiled window minus unprofiled twin, {o['source']}): {o['profiled_ms_per_round']:.2f} - "
                 f"{o['unprofiled_ms_per_round']:.2f} = {o['overhead_ms']:+.2f} ms/round ({o['overhead_pct']:+.1f}%); it lands on STEP "
                 'and the host gaps, not on kernel durations.')
    L.append('')
    s = t['sub']
    L.append('Sub-rows (ms/round): ' + '; '.join(
        f"{k} target {v['target_ms']:.2f} / draft {v['draft_ms']:.2f}" for k, v in s.items() if 'target_ms' in v))
    if 'STEP.gap_split' in s:
        g = s['STEP.gap_split']
        L.append('Host gap split (llama, summed over the round\'s inter-phase gaps, ms): ' +
                 ', '.join(f'{k} {v:.2f}' for k, v in g.items()))
    L.append('AllReduce: ' + '; '.join(f"{n}: {a['calls']:.0f} calls x {a['floor_us_per_call']:.1f} us floor at {a['payload_kib']:.0f} KiB"
                                       for n, a in t['ar'].items()))
    L += ['', 'Top kernels GPU0 (ms/round, calls/round): ' + '; '.join(
        f"{k['kernel']} {k['ms_per_round']:.2f} ({k['calls_per_round']:.0f})" for k in t['top_kernels_gpu0'][:8]), '']
    for w in t.get('warnings', []):
        L += [f'WARNING: {w}', '']
    return '\n'.join(L)


def prefill_md(t, title):
    L = [f'### {title}', '']
    L.append(f"engine {t['engine']}, {t['tokens']} tokens in {t['window_ms'] / 1e3:.1f} s of GPU0 window = {t['tok_per_s']:.0f} tok/s "
             f"({t['ms_per_1k_tokens']:.0f} ms per 1k tokens), batch {t['batch_tokens']} tokens")
    L += ['', '| class | ms | share % | kernel-sum share % | Q1 % | Q2 % | Q3 % | Q4 % |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in PRE_CLASSES:
        qs = ' | '.join(f"{q['share_pct'].get(c, 0):.1f}" for q in t['quarters'])
        L.append(f"| {c} | {t['ms_by_class'].get(c, 0):.0f} | {t['share_pct'].get(c, 0):.1f} | "
                 f"{t['kernel_sum_share_pct'].get(c, 0):.1f} | {qs} |")
    L.append('')
    L.append('Quarter midpoints (tokens of context): ' + ', '.join(str(q['approx_depth_tokens']) for q in t['quarters']) +
             '. GEMM = weight GEMV/GEMM (incl. MMQ); GDN = gated-delta-net/conv/chunk kernels; ATTN = flash attention + KV writes.')
    f = t['floors_ms']
    L += ['', f"Floors: weight re-stream {f['weight_restream']:.0f} ms ({t['floor_notes']['weight_restream']}); compute "
              f"{f['compute']:.0f} ms ({t['floor_notes']['compute']}). Headroom vs the larger floor: {t['headroom_ms']:.0f} ms "
              f"({t['headroom_pct_of_window']:.0f}% of the window).", '',
          'Top kernels GPU0 (ms, calls): ' + '; '.join(f"{k['kernel']} {k['ms']:.0f} ({k['calls']})" for k in t['top_kernels_gpu0'][:8]), '']
    L += [f'WARNING: {w}' for w in t.get('warnings', [])]
    return '\n'.join(L)


LABELS = ("Labels: M = read from the trace; D = derived (arithmetic on trace values, model sizes or published constants); "
          "G = guess. Axis times are M (priority partition of GPU0's timeline); floors are D (bytes / (tp*448 GB/s), AR table "
          "10 KiB 9 us .. 40 KiB 14 us, best demonstrated, extrapolated above); the prefill compute floor uses a G peak.")


def write_pack(results, out_dir, header):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'budget.json').write_text(json.dumps({'schema': 1, **header, 'traces': results}, indent=1))
    L = [f"# Attribution budget: {header.get('title', '')}", '', LABELS, '']
    for key in ('model', 'engine', 'generated_note'):
        if header.get(key):
            L.append(f'- {key}: {header[key]}')
    L.append('')
    for r in results:
        if r['status'] != 'ok':
            L += [f"### {r['id']}: REJECTED", '', f"Reason: {r['reason']}", '']
            continue
        t = r['result']
        L.append(decode_md(t, r['id']) if t['kind'] == 'decode' else prefill_md(t, r['id']))
    groups = collections.defaultdict(list)
    for r in results:
        if r['status'] == 'ok' and r['result']['kind'] == 'decode' and r.get('group'):
            groups[r['group']].append(r)
    for gname, dec in groups.items():
        if len(dec) < 2:
            continue
        dec.sort(key=lambda r: r['result']['depth'])
        a, b = dec[0]['result'], dec[-1]['result']
        L += [f"### Depth scaling {a['depth']} -> {b['depth']} (group {gname}: same build and flags)", '', '| axis | ms/round at ' + str(a['depth']) + ' | ms/round at ' +
              str(b['depth']) + ' | delta ms |', '|---|---:|---:|---:|']
        for ax in AXES:
            x, y = a['axes'][ax]['measured_ms'], b['axes'][ax]['measured_ms']
            L.append(f'| {ax} | {x:.2f} | {y:.2f} | {y - x:+.2f} |')
        L.append(f"| round | {a['round_ms_gpu0']:.2f} | {b['round_ms_gpu0']:.2f} | {b['round_ms_gpu0'] - a['round_ms_gpu0']:+.2f} |")
        L.append('')
    (out / 'budget.md').write_text('\n'.join(L) + '\n')


def load_model(path):
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = HERE / 'models' / path
    return json.loads(p.read_text())


def run_one(spec, model, defaults):
    kind = spec.get('kind', 'decode')
    try:
        eng0 = spec.get('engine') or defaults['engine']
        tr = Trace(spec['sqlite'], need_memcpy=(kind == 'decode' and eng0 == 'llama'), need_runtime=(kind == 'decode'))
        engine = spec.get('engine') or defaults['engine'] or detect_engine(tr)
        if kind == 'decode':
            res = decode_budget(tr, model, spec['depth'], engine, spec.get('k', defaults.get('k')),
                                spec.get('min_rounds', defaults.get('min_rounds', 30)), spec.get('trim'),
                                spec.get('tok_per_round'), spec.get('draft_head_bytes'), spec.get('skip_ms', 750.0),
                                spec.get('unprofiled'))
        else:
            res = prefill_budget(tr, model, engine, spec['tokens'], window_ms=spec.get('window_ms'),
                                 batch_tokens=spec.get('batch_tokens'))
        res['trace'] = {'path': str(spec['sqlite']), 'sha': tr.sha(), 'span_ms': (tr.t1 - tr.t0) / 1e6}
        return {'id': spec['id'], 'status': 'ok', 'group': spec.get('group'), 'result': res}
    except Reject as e:
        return {'id': spec['id'], 'status': 'rejected', 'reason': str(e)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('decode', 'prefill'):
        p = sub.add_parser(name)
        p.add_argument('sqlite')
        p.add_argument('--model', required=True, help='models/*.json name or path')
        p.add_argument('--engine', choices=['llama', 'hq'])
        p.add_argument('--out', required=True)
        p.add_argument('--id')
        if name == 'decode':
            p.add_argument('--depth', type=int, required=True)
            p.add_argument('--k', type=int)
            p.add_argument('--min-rounds', type=int, default=30)
            p.add_argument('--trim', type=float)
            p.add_argument('--tok-per-round', type=float)
            p.add_argument('--draft-head-bytes', type=int)
        else:
            p.add_argument('--tokens', type=int, required=True)
            p.add_argument('--batch-tokens', type=int)
            p.add_argument('--window-ms', help='a:b relative to the first kernel')
    p = sub.add_parser('pack', help='manifest.json listing traces -> budget.json/budget.md')
    p.add_argument('manifest')
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'pack':
        man = json.loads(Path(a.manifest).read_text())
        model = load_model(man['model'])
        defaults = {'engine': man.get('engine') or model.get('engine'), 'k': man.get('k'), 'min_rounds': man.get('min_rounds', 30)}
        results = [run_one(t, model, defaults) for t in man['traces']]
        header = {'title': man.get('title', ''), 'model': model['name'], 'engine': defaults['engine']}
        write_pack(results, a.out, header)
        bad = [r for r in results if r['status'] != 'ok']
        for r in bad:
            print(f"REJECTED {r['id']}: {r['reason']}", file=sys.stderr)
        print(f"{len(results) - len(bad)}/{len(results)} traces accepted -> {a.out}/budget.md")
        return 1 if bad else 0
    model = load_model(a.model)
    spec = {'id': a.id or Path(a.sqlite).stem, 'sqlite': a.sqlite, 'kind': a.cmd, 'engine': a.engine}
    if a.cmd == 'decode':
        spec.update(depth=a.depth, k=a.k, min_rounds=a.min_rounds, trim=a.trim, tok_per_round=a.tok_per_round,
                    draft_head_bytes=a.draft_head_bytes)
    else:
        spec.update(tokens=a.tokens, batch_tokens=a.batch_tokens)
        if a.window_ms:
            x, y = a.window_ms.split(':')
            spec['window_ms'] = (float(x), float(y))
    r = run_one(spec, model, {'engine': a.engine or model.get('engine')})
    write_pack([r], a.out, {'title': spec['id'], 'model': model['name'], 'engine': a.engine or model.get('engine')})
    if r['status'] != 'ok':
        print('REJECTED:', r['reason'], file=sys.stderr)
        return 1
    print((decode_md if a.cmd == 'decode' else prefill_md)(r['result'], spec['id']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
