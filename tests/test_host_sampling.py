"""host_sampling.py on a synthetic nsys export built from the real nsys 2025.6.3 schema (tests/fixtures/nsys_schema.sql).

Layout (ns, 50 us grid): 60 rounds of V + 5 D blocks on two devices; every inter-block gap has the pack's four segments
(readback / cpu_pre / inputs / launch_lat). The decode thread is sampled on a global 0.5 ms grid (never on a boundary) and the
label of a sample is chosen by where it falls, so the expected ms/round per function is computed by the builder, not by the
code under test. Negative controls: samples while GPU0 runs a kernel, samples of a non-launching thread, and samples
outside the steady rounds must not be counted; a trace without sample tables, with too few samples, or whose samples are
all on a non-launching thread is rejected.
Run: CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_host_sampling
"""
import collections
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK))
import attrib_pack as ap  # noqa: E402
import host_sampling as hs  # noqa: E402

SCHEMA = PACK / 'tests/fixtures/nsys_schema.sql'
REAL = PACK / 'runs/attrib-llama/full-a1/decode-8k.sqlite'
MODEL = ap.load_model('llama-swift15-iq4xs.json')
HEAD = MODEL['head_type']
MS = 1_000_000
PID = 5 << 24
DEC, OTHER = PID + 7, PID + 9            # globalTid = pid key + thread index (the pack keys on (tid >> 24) << 24)
K, ROUNDS = 5, 60
GRID = MS // 2                           # sample spacing
V_KERNELS = [('gated_delta_net_cuda<>', MS), (f'mul_mat_vec_q<(ggml_type)2, (int)4, (bool)0>', 2 * MS)]
D_KERNEL = (f'void mul_mat_vec_q<(ggml_type){HEAD}, (int)1, (bool)0, (bool)0, (bool)0>(const void *)', MS)
GAP = {'V': 2 * MS, 'D': MS}             # host work between blocks
LAT = MS // 20                           # launch latency: GPU starts this long after the first launch call

CUDA = ('0x7f0', '/usr/lib/libcuda.so.1', 1)
GGML_CUDA = ('ggml_backend_cuda_synchronize', '/b/libggml-cuda.so.0', 0)
GGML_BASE = ('ggml_backend_sched_split_graph', '/b/libggml-base.so.0', 0)
LLAMA = ('llama_context::graph_compute', '/b/libllama.so.0', 0)
COMMON = ('common_sampler_sample', '/b/libllama-common.so.0', 0)
SLOT = ('server_context::update_slots', '/b/libllama-server-impl.so', 0)
MAIN = ('main', '/b/weekend-srv', 0)
LIBC = ('memcpy', '/usr/lib/libc.so.6', 0)
BROKEN = ('[Broken backtraces]', '[Broken backtraces]', 0)
CHAINS = {        # (segment, parity) -> stack leaf first
    ('readback', 0): [CUDA, GGML_CUDA, LLAMA, SLOT, MAIN],
    ('readback', 1): [CUDA, GGML_CUDA, LLAMA, SLOT, MAIN],
    ('cpu_pre', 0): [GGML_BASE, LLAMA, SLOT, MAIN],
    ('cpu_pre', 1): [LIBC, COMMON, SLOT, MAIN],
    ('inputs', 0): [('llama_kv_cache::set_input_k_idxs', '/b/libllama.so.0', 0), SLOT, MAIN],
    ('inputs', 1): [('llama_kv_cache::set_input_k_idxs', '/b/libllama.so.0', 0), SLOT, MAIN],
    ('launch_lat', 0): [CUDA, ('ggml_backend_cuda_graph_compute', '/b/libggml-cuda.so.0', 0), LLAMA, SLOT, MAIN],
    ('launch_lat', 1): [CUDA, ('ggml_backend_cuda_graph_compute', '/b/libggml-cuda.so.0', 0), LLAMA, SLOT, MAIN],
}
BUSY = [('ggml_cuda_op_mul_mat_vec_q', '/b/libggml-cuda.so.0', 0), LLAMA, SLOT, MAIN]      # sampled while a kernel runs
OTHER_CHAIN = [('httplib::Server::process', '/b/libllama-server-impl.so', 0), MAIN]
WARM = [('warmup_only_fn', '/b/libllama.so.0', 0), MAIN]
BARE = [('bare_leaf', '/usr/lib/libc.so.6', 0), BROKEN]                                      # no llama/ggml frame


def ddl(db):
    db.executescript(SCHEMA.read_text())


def insert(db, table, **vals):
    """Row with every NOT NULL column the caller did not set filled with 0 (the real tables have ~20 of them)."""
    cols = {r[1]: r[3] for r in db.execute(f'pragma table_info({table})')}
    row = {c: vals.get(c, 0 if nn else None) for c, nn in cols.items()}
    db.execute(f'insert into {table} ({",".join(row)}) values ({",".join("?" * len(row))})', list(row.values()))


class Builder:
    def __init__(self, path, samples=True, chain_for_samples=True, decode_tid=DEC, warm=True, bare_every=0, rounds=ROUNDS):
        self.path, self.sid, self.cid = path, {}, 0
        self.db = sqlite3.connect(path)
        ddl(self.db)
        self.expected_first = collections.Counter()
        self.expected_top = collections.Counter()
        self.expected_seg = collections.Counter()
        self.n_busy = self.n_other = self.n_warm = self.n_step = 0
        self.rounds_t = []
        self.t = 10 * MS
        for r in range(rounds + 1):               # the last V only closes the final round
            self.round(r < rounds)
        self.end = self.t
        if samples:
            self.sample(decode_tid, chain_for_samples, warm, bare_every)
        self.db.commit()

    def string(self, s):
        if s not in self.sid:
            self.sid[s] = len(self.sid) + 1
            self.db.execute('insert into StringIds values (?, ?)', (self.sid[s], s))
        return self.sid[s]

    def block(self, kernels, gap, h2d_at_end=True):
        """Launch calls at self.t, kernels from t + LAT back to back on both devices; returns (kernel end, next launch time)."""
        t = self.t
        self.cid += 1
        for n, (name, dur) in enumerate(kernels):
            self.cid += 1
            insert(self.db, 'CUPTI_ACTIVITY_KIND_RUNTIME', start=t + n * 1000, end=t + n * 1000 + 500, eventClass=0, globalTid=DEC,
                   correlationId=self.cid, nameId=self.string('cudaLaunchKernel_v7000'), returnValue=0)
            ks = t + LAT + sum(d for _, d in kernels[:n])
            for dev in (0, 1):
                insert(self.db, 'CUPTI_ACTIVITY_KIND_KERNEL', start=ks, end=ks + dur, deviceId=dev, correlationId=self.cid,
                       globalPid=PID, demangledName=self.string(name), shortName=self.string(name))
        e = t + LAT + sum(d for _, d in kernels)
        self.cid += 1
        # readback: sync returns 0.5 ms after the last kernel; inputs: H2D copy 0.3 ms before the next launch
        insert(self.db, 'CUPTI_ACTIVITY_KIND_RUNTIME', start=e + MS // 10, end=e + MS // 2, eventClass=0, globalTid=DEC,
               correlationId=self.cid, nameId=self.string('cudaStreamSynchronize_v3020'), returnValue=0)
        nxt = e + gap
        self.cid += 1
        insert(self.db, 'CUPTI_ACTIVITY_KIND_RUNTIME', start=nxt - 3 * MS // 10, end=nxt - 2 * MS // 10, eventClass=0, globalTid=DEC,
               correlationId=self.cid, nameId=self.string('cudaMemcpyAsync_v3020'), returnValue=0)
        insert(self.db, 'CUPTI_ACTIVITY_KIND_MEMCPY', start=nxt - 3 * MS // 10, end=nxt - 2 * MS // 10, deviceId=0, correlationId=self.cid,
               globalPid=PID, bytes=64, copyKind=1)
        self.gaps.append((e, nxt, kernels is V_KERNELS))
        self.busy.append((t + LAT, e))
        self.t = nxt

    def round(self, real):
        self.gaps, self.busy = [], []
        t0 = self.t
        self.block(V_KERNELS, GAP['V'])
        if real:
            for _ in range(K):
                self.block([D_KERNEL], GAP['D'])
        self.rounds_t.append((t0, self.t))
        self.all_gaps = getattr(self, 'all_gaps', []) + [g for g in self.gaps]
        self.all_busy = getattr(self, 'all_busy', []) + self.busy

    def segment_of(self, t):
        """Segment of a STEP timestamp from the closed-form layout (independent of the code under test)."""
        for e, nxt, is_v in self.all_gaps:
            if e <= t < nxt + LAT:
                if t < e + MS // 2:
                    return 'readback'
                if t < nxt - 3 * MS // 10:
                    return 'cpu_pre'
                if t < nxt:
                    return 'inputs'
                return 'launch_lat'
        return None

    def busy_at(self, t):
        return any(a <= t < b for a, b in self.all_busy)

    def chain(self, db, eid, frames):
        for depth, (sym, mod, unres) in enumerate(frames):
            insert(db, 'SAMPLING_CALLCHAINS', id=eid, symbol=self.string(sym), module=self.string(mod), unresolved=unres, stackDepth=depth)

    def sample(self, tid, chains, warm, bare_every):
        db = self.db
        insert(db, 'ThreadNames', nameId=self.string('llama-server-main'), globalTid=DEC)
        insert(db, 'ThreadNames', nameId=self.string('http-worker'), globalTid=OTHER)
        for k_, v in (('LINUX_PERF_OPTIONS:SAMPLING_PERIOD', '2336904'), ('LINUX_PERF_OPTIONS:SAMPLES_PER_BACKTRACE', '1'),
                      ('USE_DWARF', 'true'), ('USE_FRAME_POINTERS', 'false')):
            db.execute('insert into META_DATA_CAPTURE values (?, ?)', (k_, v))
        first_round, last_round = self.rounds_t[1][0] + LAT, self.rounds_t[-2][1] + LAT     # the pack drops round 0 (host_s < first kernel)
        eid, n_step = 0, 0
        t = first_round - 8 * MS + GRID // 2
        while t < self.end + 3 * MS:
            eid += 1
            seg = self.segment_of(t)
            inside = first_round <= t < last_round
            frames, kind = None, None
            if inside and seg:
                par = (t // GRID) % 2
                frames, kind = CHAINS[(seg, par)], 'step'
                n_step += 1
                if bare_every and n_step % bare_every == 0:
                    frames, kind = BARE, 'bare'
                self.expected_seg[seg] += 1
                self.n_step += 1
                lead = next((f for f in frames if hs.is_llama(f[1])), None)
                self.expected_first[lead[0] if lead else hs.NO_LLAMA] += 1
                self.expected_top[frames[0][0] if frames[0][2] == 0 else '[libcuda.so.1]'] += 1
            elif inside:
                frames = BUSY
                self.n_busy += 1
            else:
                frames = WARM
                self.n_warm += 1
            insert(db, 'COMPOSITE_EVENTS', id=eid, start=t, cpu=1, threadState=1, globalTid=tid, cpuCycles=1)
            if chains:
                self.chain(db, eid, frames)
            if seg and inside:       # a second thread samples inside every gap at the same instants: must stay out of the table
                eid += 1
                insert(db, 'COMPOSITE_EVENTS', id=eid, start=t + 1000, cpu=2, threadState=1, globalTid=OTHER, cpuCycles=1)
                self.chain(db, eid, OTHER_CHAIN)
                self.n_other += 1
            t += GRID


def manifest(path, depth=8192, **kw):
    spec = {'id': f'llama-decode-{depth}', 'sqlite': str(path), 'kind': 'decode', 'depth': depth, 'group': 'depth', 'k': K}
    spec.update(kw)
    return {'title': 'synthetic', 'model': 'llama-swift15-iq4xs.json', 'engine': 'llama', 'k': K, 'min_rounds': 40, 'traces': [spec]}


def attribute(path, **kw):
    spec = manifest(path)['traces'][0]
    tr = ap.Trace(path, need_memcpy=True, need_runtime=True)
    return hs.host_attribution(tr, MODEL, spec, {'engine': 'llama', 'k': K, 'min_rounds': 40}, **kw)


class Fixture(unittest.TestCase):
    def test_ddl_snapshot_matches_the_real_attrib_llama_trace(self):
        if not REAL.exists():
            self.skipTest('recorded attrib-llama trace missing')
        real = sqlite3.connect(REAL.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        snap = sqlite3.connect(':memory:')
        ddl(snap)
        for t in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET',
                  'StringIds', 'ThreadNames', 'META_DATA_CAPTURE'):
            self.assertEqual(real.execute('select sql from sqlite_master where name=?', (t,)).fetchone()[0].rstrip(),
                             snap.execute('select sql from sqlite_master where name=?', (t,)).fetchone()[0].rstrip(), t)
        self.assertEqual({r[0] for r in real.execute("select name from sqlite_master where type='table'")} &
                         {'COMPOSITE_EVENTS', 'SAMPLING_CALLCHAINS'}, set())      # --sample=none: why the sampling DDL comes from another capture


class Attribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.path = Path(cls.dir.name) / 'syn.sqlite'
        cls.b = Builder(cls.path, bare_every=11)
        cls.r = attribute(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def rows(self, field):
        return {r['fn']: r for r in self.r[field]}

    def test_step_equals_the_pack_and_the_layout(self):
        r = self.r
        self.assertEqual(r['n_rounds'], ROUNDS - 1)      # round 0 starts its host launches before the first kernel and is dropped
        # per round: V gap 2.0 + 5 D gaps 1.0 + 6 launch latencies 0.05 ms
        self.assertAlmostEqual(r['step_ms_per_round'], 2.0 + 5 * 1.0 + 6 * 0.05, places=6)
        # the 0.1 ms H2D input copy of each of the 6 blocks sits inside its gap and is the pack's COPY class
        self.assertAlmostEqual(r['step_ms_by_class']['COPY'], 6 * 0.1, places=6)
        self.assertAlmostEqual(r['step_ms_by_class']['IDLE'], 2.0 + 5 * 1.0 + 6 * 0.05 - 6 * 0.1, places=6)
        self.assertEqual(r['step_ms_by_class']['SAMP'], 0.0)
        self.assertAlmostEqual(r['sample_ms'], 0.5, places=9)

    def test_function_tables_match_the_builder(self):
        n, ms = self.r['n_rounds'], 0.5
        first, top = self.rows('first_llama_frame'), self.rows('top_of_stack')
        self.assertEqual(self.r['step_samples'], self.b.n_step)
        for fn, c in self.b.expected_first.items():
            self.assertEqual(first[fn]['samples'], c, fn)
            self.assertAlmostEqual(first[fn]['ms_per_round'], c * ms / n, places=9, msg=fn)
        for fn, c in self.b.expected_top.items():
            self.assertEqual(top[fn]['samples'], c, fn)
        self.assertEqual(sum(r['samples'] for r in self.r['first_llama_frame']), self.b.n_step)
        self.assertIn('[libcuda.so.1]', top)                     # unresolved leaf frames collapse to their module
        self.assertGreater(first[hs.NO_LLAMA]['samples'], 0)     # BARE stacks: counted, not dropped
        self.assertEqual(first['common_sampler_sample']['module'], 'libllama-common.so.0')

    def test_segments_split_the_samples(self):
        seg = self.r['segments']
        for name, c in self.b.expected_seg.items():
            self.assertEqual(seg[name]['samples'], c, name)
        self.assertEqual(sum(v['samples'] for v in seg.values()), self.b.n_step)
        # cpu_pre gap wall: V 1.2 ms + 5 x 0.2 ms
        self.assertAlmostEqual(self.r['segment_wall_ms_per_round']['cpu_pre'], 1.2 + 5 * 0.2, places=6)
        self.assertAlmostEqual(self.r['segment_wall_ms_per_round']['inputs'], 6 * 0.3, places=6)

    def test_negative_control_samples_during_kernels_are_not_counted(self):
        self.assertGreater(self.b.n_busy, 100)
        self.assertNotIn('ggml_cuda_op_mul_mat_vec_q', {r['fn'] for r in self.r['top_of_stack']})
        self.assertNotIn('ggml_cuda_op_mul_mat_vec_q', {r['fn'] for r in self.r['inclusive_llama_frames']})

    def test_negative_control_other_threads_and_samples_outside_the_rounds(self):
        self.assertGreater(self.b.n_other, 100)
        self.assertGreater(self.b.n_warm, 10)
        names = {r['fn'] for r in self.r['top_of_stack']} | {r['fn'] for r in self.r['first_llama_frame']}
        self.assertFalse(names & {'httplib::Server::process', 'warmup_only_fn'})
        self.assertEqual([t['name'] for t in self.r['decode_threads']], ['llama-server-main'])
        self.assertEqual(self.r['other_threads_samples_in_window'][0]['thread'], 'http-worker')
        self.assertEqual(self.r['other_threads_samples_in_window'][0]['samples_in_window'], self.b.n_other)

    def test_naive_attribution_would_differ(self):
        """The fixture discriminates: charging every decode-thread sample in the window (what the STEP filter prevents) is wrong."""
        db = sqlite3.connect(self.path)
        a, b = self.b.rounds_t[1][0] + LAT, self.b.rounds_t[-2][1] + LAT
        n_all = db.execute('select count(*) from COMPOSITE_EVENTS where globalTid=? and start>=? and start<?', (DEC, a, b)).fetchone()[0]
        self.assertGreater(n_all, 1.5 * self.r['step_samples'])
        self.assertEqual(n_all, self.b.n_step + self.b.n_busy)
        self.assertEqual(self.r['decode_threads'][0]['samples_in_window'], n_all)     # warm-up and round-0 samples are outside the window

    def test_coverage_and_unsampled_time(self):
        r = self.r
        self.assertAlmostEqual(r['sampled_cpu_ms_per_round'] + r['unsampled_ms_per_round'], r['step_ms_per_round'], places=9)
        self.assertAlmostEqual(r['coverage_of_step_wall'], 1.0, delta=0.03)
        self.assertEqual(r['capture']['USE_DWARF'], 'true')

    def test_depth_delta_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            a, b = d / 'a.sqlite', d / 'b.sqlite'
            Builder(a, rounds=50)
            Builder(b, rounds=50, bare_every=2)         # the "deep" trace: half of the STEP samples lose their llama/ggml frame
            m = manifest(a, 8192)
            m['traces'].append(manifest(b, 90000)['traces'][0])
            (d / 'm.json').write_text(json.dumps(m))
            r = subprocess.run([sys.executable, str(PACK / 'host_sampling.py'), str(d / 'm.json'), '--out', str(d / 'out')],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            md = (d / 'out/host.md').read_text()
            js = json.loads((d / 'out/host.json').read_text())
            self.assertEqual([t['depth'] for t in js['traces']], [8192, 90000])
            self.assertIn('### Depth delta 8192 -> 90000', md)
            self.assertIn('### Sampling overhead', md)
            row = next(x for x in md.split('### Depth delta')[1].splitlines() if x.startswith(f'| `{hs.NO_LLAMA}`'))
            x, y, dl = [float(v) for v in row.strip('|').split('|')[-3:]]
            self.assertEqual(x, 0.0)
            self.assertGreater(y, 0.5)
            self.assertAlmostEqual(dl, y - x, places=3)


class Rejections(unittest.TestCase):
    def build(self, d, **kw):
        p = Path(d) / 'x.sqlite'
        Builder(p, **kw)
        return p

    def test_no_sample_tables(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.build(d)
            db = sqlite3.connect(p)
            db.execute('drop table SAMPLING_CALLCHAINS')
            db.commit()
            db.close()
            with self.assertRaisesRegex(ap.Reject, 'no CPU samples'):
                attribute(p)

    def _no_samples(self, d):
        p = self.build(d)
        db = sqlite3.connect(p)
        db.execute('delete from COMPOSITE_EVENTS')
        db.execute('delete from SAMPLING_CALLCHAINS')
        db.commit()
        db.close()
        return p

    def test_no_samples_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ap.Reject, r'only 0 decode-thread CPU samples'):
                attribute(self._no_samples(d))

    def test_samples_only_on_a_thread_that_never_launches(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ap.Reject, r'only 0 decode-thread CPU samples'):
                attribute(self.build(d, decode_tid=OTHER + 1))

    def test_too_few_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.build(d)
            with self.assertRaisesRegex(ap.Reject, r'only \d+ decode-thread CPU samples inside STEP'):
                attribute(p, min_samples=10 ** 6)

    def test_samples_without_callchains_are_counted_as_unresolved(self):
        with tempfile.TemporaryDirectory() as d:
            r = attribute(self.build(d, chain_for_samples=False))
            self.assertEqual(r['first_llama_frame'][0]['fn'], hs.NO_LLAMA)
            self.assertTrue(any('no llama/ggml frame' in w for w in r['warnings']))

    def test_cli_exit_code_follows_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._no_samples(d)
            (Path(d) / 'm.json').write_text(json.dumps(manifest(p)))
            r = subprocess.run([sys.executable, str(PACK / 'host_sampling.py'), str(Path(d) / 'm.json'), '--out', str(Path(d) / 'o')],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn('REJECTED llama-decode-8192', r.stderr)
            self.assertIn('REJECTED', (Path(d) / 'o/host.md').read_text())


class RealChains(unittest.TestCase):
    """tests/fixtures/real_chains.sqlite: 439 samples (every 25th) of the main thread of the frozen stack2 server decoding on the CPU, captured
    with nsys 2025.6.3 `--sample=process-tree --backtrace=dwarf --sampling-period=2336904 --samples-per-backtrace=1`."""
    PATH = PACK / 'tests/fixtures/real_chains.sqlite'
    TID = 333381773505918

    def test_real_stacks_are_labelled_by_leaf_and_by_first_llama_frame(self):
        ev, chains, other, names, cfg = hs.load_samples(self.PATH, [self.TID], 0, 1 << 62)
        self.assertEqual((len(ev), len(chains), dict(other)), (439, 439, {}))
        self.assertEqual((cfg['USE_DWARF'], cfg['LINUX_PERF_OPTIONS:SAMPLES_PER_BACKTRACE']), ('true', '1'))
        top, first, broken = collections.Counter(), collections.Counter(), 0
        for eid, t, tid in ev:
            tl, fl, inc, b = hs.classify_chain(chains[eid])
            top[tl[0]] += 1
            first[fl[0]] += 1
            broken += b
            self.assertNotIn(hs.BROKEN, (tl[0], fl[0]))                # the sentinel frame is never a label
            self.assertNotEqual(fl[0], hs.NO_LLAMA)
            self.assertTrue(hs.is_llama(fl[1]))
        self.assertEqual(broken, 439)                                   # outer frames beyond the dwarf stack slice are lost (documented limit)
        self.assertEqual(top.most_common(1)[0][0], 'ggml_gemv_q4_K_8x8_q8_K')
        self.assertIn('[libgomp.so.1.0.0]', top)                        # unresolved leaf collapses to its module
        self.assertEqual(first['ggml_graph_compute_thread'], 24)        # leaf in libgomp/libc: first llama/ggml frame is its caller

    def test_negative_control_leaf_only_stacks_lose_the_llama_frame(self):
        """Why every sample needs its chain (spb 1, dwarf): with the leaf alone, samples whose leaf is libgomp/libc have no llama/ggml frame."""
        ev, chains, *_ = hs.load_samples(self.PATH, [self.TID], 0, 1 << 62)
        non_llama_leaf = [eid for eid, t, tid in ev if not hs.is_llama(chains[eid][0][1])]
        self.assertGreater(len(non_llama_leaf), 20)
        for eid in non_llama_leaf:
            self.assertNotEqual(hs.classify_chain(chains[eid])[1][0], hs.NO_LLAMA)
            self.assertEqual(hs.classify_chain(chains[eid][:1])[1][0], hs.NO_LLAMA)


class RealTrace(unittest.TestCase):
    def test_attrib_llama_trace_has_no_samples_and_is_rejected(self):
        """The existing CUDA-only decode-8k trace (--sample=none) must be refused, not summarised."""
        if not REAL.exists():
            self.skipTest('recorded attrib-llama trace missing')
        tr = ap.Trace(REAL, need_memcpy=True, need_runtime=True)
        with self.assertRaisesRegex(ap.Reject, 'no CPU samples'):
            hs.host_attribution(tr, MODEL, manifest(REAL)['traces'][0], {'engine': 'llama', 'k': K, 'min_rounds': 40})


if __name__ == '__main__':
    unittest.main()
