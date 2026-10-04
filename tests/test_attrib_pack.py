"""Tests on real recorded nsys exports (skipped when a trace is absent) plus negative controls.

Run: CUDA_VISIBLE_DEVICES= nice -n 19 python3 -m unittest -v   (tests/__init__.py makes plain discovery work)
Published reference values: SP/gpu-sampling/items/96b-attrib.md (llama P/B traces), SP/queue/done/47 + attrib-item47-np1-P.out,
~/repos/draft-anatomy/codex/REPORT.md (HQ idle-run: 197 steps, weight+head 17.664+2.026 ms).
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import attrib_pack as ap  # noqa: E402

SP = Path('/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad')
NP1 = SP / 'queue/out/np2-nsys/np1-k3-d8192.sqlite'
PTHINK = SP / 'queue/out/gpu-sampling-b/P-think.sqlite'
BTOOL = SP / 'queue/out/gpu-sampling-b/B-tool.sqlite'
HQ = SP / 'gpu0-display/idle-run/profile/decode8k.sqlite'
RT = Path.home() / 'repos/round-trace/llama.cpp/codex/runs/20261001-214325/trace/capture.sqlite'
LL = ap.load_model('llama-swift15-iq4xs.json')
HQM = ap.load_model('hq-swift15-syvfast.json')
HQC = ap.load_model('hq-swift15-codexh.json')
FULL_HEAD = 1042964480  # the 2026-09 traces predate the draft-vocab subset


def need(*paths):
    return unittest.skipUnless(all(p.exists() for p in paths), 'recorded trace missing: ' + ', '.join(map(str, paths)))


def llama(path, **kw):
    kw.setdefault('min_rounds', 5)
    return ap.decode_budget(ap.Trace(path, need_memcpy=True), LL, 8192, 'llama', k=3, draft_head_bytes=FULL_HEAD, **kw)


def mutate(src, sql, tmp):
    dst = Path(tmp) / 'mut.sqlite'
    shutil.copy(src, dst)
    db = sqlite3.connect(dst)
    for stmt in sql:
        db.execute(stmt)
    db.commit()
    db.execute('vacuum')
    db.close()
    return dst


class Reproduces(unittest.TestCase):
    @need(NP1)
    def test_item47_np1(self):
        r = llama(NP1, min_rounds=30)
        self.assertEqual(r['n_rounds'], 70)                      # 96b-attrib.md: item-47 TRIM set N=70
        self.assertAlmostEqual(r['round_ms_host'], 38.36, delta=0.01)
        self.assertAlmostEqual(r['sub']['FWD.lm_head']['draft_ms'], 3.70, delta=0.02)  # published draft lm_head 3.70

    @need(PTHINK, BTOOL)
    def test_96b_traces(self):
        p = llama(PTHINK)
        self.assertEqual((p['n_rounds'], round(p['round_ms_host'], 2)), (7, 35.24))
        b = llama(BTOOL)
        self.assertEqual((b['n_rounds'], round(b['round_ms_host'], 2)), (6, 34.50))
        # published KV 0.90 ms counted flash-attn + GDN kernels only; GDN state concat/get_rows/cpy (L2) add 0.88 ms
        self.assertAlmostEqual(b['sub']['KV_ATTN']['target_ms'] + b['sub']['KV_GDN']['target_ms'], 0.90 + 0.88, delta=0.1)

    @need(HQ)
    def test_hq_matches_anatomy(self):
        tr = ap.Trace(HQ)
        r = ap.decode_budget(tr, HQM, 8192, 'hq', k=3, min_rounds=30)
        self.assertEqual(r['n_rounds'], 197)                     # anatomy REPORT: 197 retained steps
        # anatomy counts Marlin + head; the 48 wmma bf16 128x1 GEMMs per step (GDN in_proj_a/b, 1.37 ms) are weight streaming too
        self.assertAlmostEqual(r['axes']['FWD']['measured_ms'], 17.664 + 2.026 + 1.37, delta=0.03)
        self.assertEqual(r['draft_forwards'], 3)


class Invariants(unittest.TestCase):
    @need(NP1)
    def test_axes_cover_the_round(self):
        r = llama(NP1, min_rounds=30)
        tot = sum(a['measured_ms'] for a in r['axes'].values())
        self.assertAlmostEqual(tot, r['round_ms_gpu0'], places=6)
        for a in r['axes'].values():
            self.assertAlmostEqual(a['target_ms'] + a['draft_ms'], a['measured_ms'], places=9)
            self.assertGreaterEqual(a['headroom_ms'], 0)

    @need(NP1)
    def test_partition_is_exact_on_random_windows(self):
        tr = ap.Trace(NP1, need_memcpy=True)
        d = tr.devs[0]
        for a in (tr.t0 + 5_000_000, tr.t0 + 1_234_567_890, tr.t1 - 40_000_000):
            part, _ = ap.partition(tr, d, a, a + 33_333_333)
            self.assertEqual(sum(part.values()), 33_333_333)

    @need(RT)
    def test_prefill_shares_sum_to_100(self):
        r = ap.prefill_budget(ap.Trace(RT), LL, 'llama', 505, min_markers=64)
        self.assertAlmostEqual(sum(r['share_pct'].values()), 100.0, places=6)
        for q in r['quarters']:
            self.assertAlmostEqual(sum(q['share_pct'].values()), 100.0, places=6)
        self.assertGreater(r['share_pct']['GEMM'], r['share_pct']['ATTN'])


class NegativeControls(unittest.TestCase):
    @need(BTOOL)
    def test_missing_kernel_table_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SP) as t:
            bad = mutate(BTOOL, ['drop table CUPTI_ACTIVITY_KIND_KERNEL'], t)
            with self.assertRaisesRegex(ap.Reject, 'missing tables'):
                ap.Trace(bad)

    @need(BTOOL)
    def test_empty_kernel_table_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SP) as t:
            bad = mutate(BTOOL, ['delete from CUPTI_ACTIVITY_KIND_KERNEL'], t)
            with self.assertRaisesRegex(ap.Reject, 'empty'):
                ap.Trace(bad)

    @need(NP1)
    def test_truncated_range_is_rejected(self):
        # Keep only the first 600 ms of a 4 s capture: complete steady rounds fall below the floor.
        with tempfile.TemporaryDirectory(dir=SP) as t:
            db = sqlite3.connect(NP1)
            t0 = db.execute('select min(start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
            db.close()
            cut = [f'delete from {tb} where start > {t0 + 600_000_000}'
                   for tb in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_MEMCPY')]
            bad = mutate(NP1, cut, t)
            with self.assertRaisesRegex(ap.Reject, r'only \d+ complete steady rounds'):
                llama(bad, min_rounds=30)

    @need(NP1)
    def test_byte_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SP) as t:
            dst = Path(t) / 'cut.sqlite'
            data = NP1.read_bytes()
            dst.write_bytes(data[:len(data) // 2])
            with self.assertRaises(ap.Reject):
                ap.Trace(dst, need_memcpy=True)

    @need(NP1)
    def test_wrong_k_is_rejected(self):
        with self.assertRaisesRegex(ap.Reject, 'no round matches'):
            ap.decode_budget(ap.Trace(NP1, need_memcpy=True), LL, 8192, 'llama', k=5, min_rounds=5)

    @need(BTOOL)
    def test_single_device_trace_is_rejected_for_tp2(self):
        with tempfile.TemporaryDirectory(dir=SP) as t:
            bad = mutate(BTOOL, ['delete from CUPTI_ACTIVITY_KIND_KERNEL where deviceId = 1'], t)
            with self.assertRaisesRegex(ap.Reject, 'device'):
                llama(bad)

    @need(NP1)
    def test_decode_trace_is_not_a_prefill_trace(self):
        with self.assertRaisesRegex(ap.Reject, 'not a prefill trace'):
            ap.prefill_budget(ap.Trace(NP1), LL, 'llama', 32768)

    def test_garbage_file_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SP) as t:
            f = Path(t) / 'x.sqlite'
            f.write_bytes(b'not a database' * 100)
            with self.assertRaises(ap.Reject):
                ap.Trace(f)

    @need(HQ)
    def test_hq_trace_with_wrong_model_bytes_warns(self):
        # weights 30% larger than the checkpoint this trace ran: a floor above the measurement must be flagged.
        bad = dict(HQM, weights={**HQM['weights'], 'target_stream_bytes': HQM['weights']['target_stream_bytes'] * 1.3})
        r = ap.decode_budget(ap.Trace(HQ), bad, 8192, 'hq', k=3, min_rounds=30)
        self.assertTrue(any('below its physical floor' in w for w in r['warnings']))
        ok = ap.decode_budget(ap.Trace(HQ), HQM, 8192, 'hq', k=3, min_rounds=30)
        self.assertFalse(any('below its physical floor' in w for w in ok['warnings']))

    @need(HQ)
    def test_hq_trace_without_graph_kernels_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SP) as t:
            bad = mutate(HQ, ['update CUPTI_ACTIVITY_KIND_KERNEL set graphId = NULL'], t)
            with self.assertRaisesRegex(ap.Reject, 'no graph-replayed kernels'):
                ap.decode_budget(ap.Trace(bad), HQM, 8192, 'hq', k=3, min_rounds=30)


FULL_A1 = Path(__file__).resolve().parent.parent / 'runs/attrib-llama/full-a1'
MMQ = 'void mul_mat_q<(ggml_type)23, 128, false>(char const*, int*)'
SYN_MS = 1_000_000


def synthetic_prefill(path, n_ubatch=20, mmq_per_ubatch=100, host_gap_ms=40.0, dead_ms=0.0, extra=()):
    """Two devices, one marker run per ubatch (kernels 0.2 ms apart) and `host_gap_ms` of host work between ubatches, as at -ub 128."""
    db = sqlite3.connect(path)
    db.execute('create table StringIds (id integer primary key, value text)')
    db.execute('create table CUPTI_ACTIVITY_KIND_KERNEL (start, end, deviceId, demangledName, correlationId, globalPid)')
    db.executemany('insert into StringIds values (?, ?)', [(1, MMQ), (2, 'rms_norm_f32')])
    rows, cid, t = [], 0, 1_000_000
    for _ in range(n_ubatch):
        for i in range(mmq_per_ubatch):
            for dev in (0, 1):
                cid += 1
                rows.append((t, t + 150_000, dev, 2 if i % 4 == 3 else 1, cid, 0))
            t += 200_000
        t += int(host_gap_ms * SYN_MS)
    for off_ms in extra:
        for dev in (0, 1):
            rows.append((t + int(off_ms * SYN_MS), t + int(off_ms * SYN_MS) + 100_000, dev, 1, 0, 0))
    db.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?)', rows)
    db.commit()
    db.close()


class PrefillWindow(unittest.TestCase):
    def budget(self, **kw):
        with tempfile.TemporaryDirectory(dir=SP) as t:
            f = Path(t) / 'syn.sqlite'
            synthetic_prefill(f, **kw)
            return ap.prefill_budget(ap.Trace(f, need_runtime=False), LL, 'llama', 20 * 128, min_markers=64)

    def test_host_gaps_between_ubatches_do_not_split_the_window(self):
        # Negative control: with the old 25 ms run gap the window was one ubatch (full-a1: 0.3 s of a 45.7 s prefill).
        r = self.budget()
        self.assertEqual(r['n_runs'], 1)
        span = 20 * (100 * 0.2 + 40) - 40
        self.assertGreater(r['window_ms'], span - 0.5)
        self.assertLessEqual(r['window_ms'], span)
        self.assertEqual(r['warnings'], [])
        self.assertAlmostEqual(r['tok_per_s'], 20 * 128 / r['window_ms'] * 1e3, places=6)

    def test_old_gap_threshold_would_pick_one_ubatch(self):
        with tempfile.TemporaryDirectory(dir=SP) as t:
            f = Path(t) / 'syn.sqlite'
            synthetic_prefill(f)
            r = ap.prefill_budget(ap.Trace(f, need_runtime=False), LL, 'llama', 2560, min_markers=64, run_gap_ms=25.0)
        self.assertLess(r['window_ms'], 25)
        self.assertTrue(any('outside the chosen window' in w for w in r['warnings']))

    def test_a_much_later_stray_marker_run_does_not_extend_the_window(self):
        r = self.budget(extra=[5000.0])
        self.assertEqual(r['n_runs'], 2)
        self.assertLess(r['window_ms'], 20 * (100 * 0.2 + 40))

    @unittest.skipUnless((FULL_A1 / 'prefill-32k.sqlite').exists(), 'full-a1 trace missing')
    def test_full_a1_prefill_agrees_with_the_server(self):
        # server: prompt_ms 45706 for 32769 tokens (runs.json); the profiler adds < 2%.
        runs = json.loads((FULL_A1 / 'runs.json').read_text())
        w = next(x for x in runs['windows'] if x['name'] == 'prefill-32k')
        r = ap.prefill_budget(ap.Trace(FULL_A1 / 'prefill-32k.sqlite', need_runtime=False), LL, 'llama', 32768)
        self.assertAlmostEqual(r['tok_per_s'], w['prompt_per_second'], delta=0.03 * w['prompt_per_second'])
        self.assertEqual(r['warnings'], [])


if __name__ == '__main__':
    unittest.main()
