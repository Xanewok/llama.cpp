"""CPU tests for the GPU-item plumbing: gpu/hq_drive.py against a mock vLLM server, gpu/make_manifest.py, the survivor
reap in gpu/attrib_hq.sh, and gpuq's classification of a successful run.

The mock reproduces the two behaviours the real server is known for: /stop_profile of the last range never answers
(nsys repeat-shutdown kills the server), and a streaming request is cut at that moment.
"""
import json
import os
import re
import sqlite3
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
SP = Path(os.environ.get('ATTRIB_SCRATCH', '/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/'
                         '93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad'))
ITEMS = SP / 'overnight/items'
GPUQ = Path.home() / 'repos/gpuq'


class Mock(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, ranges, die_on_last=True):
        super().__init__(('127.0.0.1', 0), Handler)
        self.events, self.open_range, self.ranges, self.die_on_last = [], False, ranges, die_on_last
        self.stopped = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}') if n else {}

    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({'data': [{'id': 'local-benchmark'}]})

    def do_POST(self):
        s, b = self.server, self.body()
        s.events.append(self.path)
        if self.path == '/tokenize':
            self.reply({'tokens': list(range(1000))})
        elif self.path == '/reset_prefix_cache':
            self.reply({})
        elif self.path == '/start_profile':
            assert not s.open_range, 'nested range'
            s.open_range = True
            self.reply({})
        elif self.path == '/stop_profile':
            assert s.open_range, 'stop without start'
            s.open_range = False
            s.stopped += 1
            if s.die_on_last and s.stopped == s.ranges:
                self.connection.close()
                return
            self.reply({})
        elif self.path == '/v1/completions' and not b.get('stream'):
            self.reply({'usage': {'prompt_tokens': len(b['prompt'])}})
        elif self.path == '/v1/completions':
            s.events.append(f"stream:{b['max_tokens']}")
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            try:
                for i in range(b['max_tokens']):
                    self.wfile.write(b'data: ' + json.dumps({'choices': [{'text': 'x'}]}).encode() + b'\n\n')
                    self.wfile.flush()
                    time.sleep(0.01)
                    if s.die_on_last and s.stopped == s.ranges:
                        return
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except OSError:
                pass


def drive(server, windows, delay='0.2', window='0.3', timeout=120):
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        with tempfile.TemporaryDirectory(dir=SP) as d:
            r = subprocess.run([sys.executable, str(PACK / 'gpu' / 'hq_drive.py'), '--url', f'http://127.0.0.1:{server.server_port}',
                                '--out', d, '--windows', windows, '--delay', delay, '--window', window,
                                '--corpus', str(PACK / 'gpu' / 'inputs' / 'hq-corpus.txt')],
                               capture_output=True, text=True, timeout=timeout)
            runs = json.loads((Path(d) / 'runs.json').read_text())
            return r, runs
    finally:
        server.shutdown()
        server.server_close()


CANARY_WINDOWS = re.search(r'^\[ "\$CANARY" = 1 \] && WINDOWS=(\S+)$', (PACK / 'gpu/attrib_hq.sh').read_text(), re.M).group(1)


class HqDriver(unittest.TestCase):
    def test_three_windows_and_last_stop_never_answers(self):
        s = Mock(3)
        r, runs = drive(s, 'prefill-32k,decode-8k,decode-90k')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual([w['name'] for w in runs['windows']], ['prefill-32k', 'decode-8k', 'decode-90k'])
        self.assertEqual(s.events.count('/start_profile'), 3)
        self.assertEqual(s.events.count('/stop_profile'), 3)
        self.assertEqual(s.events.count('/reset_prefix_cache'), 3)
        self.assertIsNone(runs['windows'][2]['stop_s'])
        self.assertTrue(all(w['stop_s'] is not None for w in runs['windows'][:2]))
        self.assertGreater(runs['windows'][0]['tokens'], 30000)

    def test_twin_runs_before_the_window_and_gives_ms_per_step(self):
        s = Mock(1)
        r, runs = drive(s, 'decode-8k')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertLess(s.events.index('stream:400'), s.events.index('/start_profile'))     # twin before the window
        ms = runs['windows'][0]['twin']['ms_per_step']
        self.assertTrue(10 < ms < 40, ms)                                                   # mock emits a chunk every ~10 ms

    def test_canary_two_windows_exercises_repeat(self):
        s = Mock(2)
        r, runs = drive(s, CANARY_WINDOWS)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(s.events.count('/start_profile'), 2)
        self.assertEqual([w['name'] for w in runs['windows']], ['prefill-32k', 'decode-8k'])
        self.assertIsNotNone(runs['windows'][0]['stop_s'])
        self.assertIsNone(runs['windows'][1]['stop_s'])

    def test_last_stop_is_capped(self):
        sys.path.insert(0, str(PACK / 'gpu'))
        import hq_drive
        self.assertLessEqual(hq_drive.LAST_STOP_S, 90)
        self.assertGreater(hq_drive.STOP_S, hq_drive.LAST_STOP_S)

    def test_negative_control_early_connection_loss_is_rejected(self):
        # the server dies on the FIRST stop although two ranges were planned: the driver must fail, not report ok
        s = Mock(1)
        r, runs = drive(s, 'decode-8k,decode-90k')
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('error', runs['windows'][-1])


def make_sqlite(path, names):
    db = sqlite3.connect(path)
    db.execute('create table StringIds(id integer primary key, value text)')
    db.execute('create table CUPTI_ACTIVITY_KIND_KERNEL(start int, end int, deviceId int, demangledName int)')
    for i, n in enumerate(names):
        db.execute('insert into StringIds values (?, ?)', (i, n))
        db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, 0, ?)', (i * 10, i * 10 + 5, i))
    db.commit()
    db.close()


class Manifest(unittest.TestCase):
    def run_mm(self, d, windows, extra_runs=None):
        d = Path(d)
        (d / 'runs.json').write_text(json.dumps({'windows': windows}))
        subprocess.run([sys.executable, str(PACK / 'gpu' / 'make_manifest.py'), '--engine', 'hq', '--model', 'hq-swift15-codexh.json',
                        '--out-dir', str(d), '--k', '5', '--title', 't', '--manifest', str(d / 'm.json')], check=True)
        return {x['id']: x for x in json.loads((d / 'm.json').read_text())['traces']}

    def test_hq_assigns_by_content_then_range_order(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            d = Path(d)
            # files written out of range order on purpose: names carry the ordinal, content tells prefill from decode
            make_sqlite(d / 'hq_2.sqlite', ['fused_sigmoid_gating_delta_rule_update_kernel'])
            time.sleep(0.02)
            make_sqlite(d / 'hq.sqlite', ['chunk_gated_delta_rule_fwd_kernel_h_blockdim64', 'BatchPrefillWithPagedKVCacheKernel'])
            make_sqlite(d / 'hq_1.sqlite', ['fused_sigmoid_gating_delta_rule_update_kernel'])
            t = self.run_mm(d, [{'name': 'prefill-32k'}, {'name': 'decode-8k', 'twin': {'ms_per_step': 31.5}}, {'name': 'decode-90k'}])
            self.assertTrue(t['hq-prefill-32k']['sqlite'].endswith('hq.sqlite'))
            self.assertTrue(t['hq-decode-8k']['sqlite'].endswith('hq_1.sqlite'))
            self.assertTrue(t['hq-decode-90k']['sqlite'].endswith('hq_2.sqlite'))
            self.assertEqual(t['hq-decode-8k']['unprofiled']['ms'], 31.5)
            self.assertNotIn('unprofiled', t['hq-decode-90k'])

    def test_canary_windows_cover_the_hq_prefill_markers_and_two_ranges(self):
        # The canary must carry a prefill window (MARKERS['hq'], assign_hq) and >= 2 ranges (repeat-shutdown).
        self.assertEqual(CANARY_WINDOWS, 'prefill-32k,decode-8k')
        with tempfile.TemporaryDirectory(dir=SP) as d:
            d = Path(d)
            make_sqlite(d / 'hq_1.sqlite', ['fused_sigmoid_gating_delta_rule_update_kernel'])
            make_sqlite(d / 'hq.sqlite', ['chunk_gated_delta_rule_fwd_kernel_h_blockdim64', 'BatchPrefillWithPagedKVCacheKernel'])
            t = self.run_mm(d, [{'name': n} for n in CANARY_WINDOWS.split(',')])
            self.assertTrue(t['hq-prefill-32k']['sqlite'].endswith('hq.sqlite'))
            self.assertTrue(t['hq-decode-8k']['sqlite'].endswith('hq_1.sqlite'))

    def test_negative_control_missing_report_rejected_by_pack(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            d = Path(d)
            make_sqlite(d / 'hq.sqlite', ['BatchPrefillWithPagedKVCacheKernel'])
            t = self.run_mm(d, [{'name': 'prefill-32k'}, {'name': 'decode-8k'}])
            self.assertIn('missing-decode-8k', t['hq-decode-8k']['sqlite'])
            r = subprocess.run([sys.executable, str(PACK / 'attrib_pack.py'), 'pack', str(d / 'm.json'), '--out', str(d / 'pack')],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertIn('REJECTED hq-decode-8k', r.stderr)


def bash_functions(script, first, last):
    text = Path(script).read_text()
    a, b = text.index(first), text.index(last)
    return text[a:b]


class SurvivorReap(unittest.TestCase):
    def test_members_recorded_while_alive_are_reaped_by_exact_pid_and_bystander_survives(self):
        funcs = bash_functions(PACK / 'gpu/attrib_hq.sh', 'session_pids()', 'stop_server() {')
        prog = f'''
log() {{ echo "$*"; }}
{funcs}
setsid bash -c 'sleep 300 & sleep 300 & wait' & W=$!
sleep 300 & BYSTANDER=$!
sleep 1
record_members "$W"
echo "recorded ${{#MEMBERS[@]}}"
reap_members
sleep 0.5
kill -0 "$BYSTANDER" 2>/dev/null && echo bystander-alive
kill -0 "$W" 2>/dev/null && echo wrapper-alive || echo wrapper-dead
kill "$BYSTANDER"
'''
        r = subprocess.run(['bash', '-c', prog], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stdout, r'recorded [3-9]')            # bash + 2 sleeps (+ the sleep 1 subshell is not a member)
        self.assertIn('bystander-alive', r.stdout)
        self.assertIn('wrapper-dead', r.stdout)
        self.assertEqual(r.stdout.count('survivor'), r.stdout.count('TERM'))


NEW_TRAP = 'trap \'rc=$?; { [ -z "$LAUNCHED" ] || [ ! -e "$HQ/serve.pid" ]; } || stop_server; reap_members; log "exit rc=$rc"\' EXIT'
OLD_TRAP = 'trap \'rc=$?; [ ! -e "$HQ/serve.pid" ] || stop_server; reap_members; log "exit rc=$rc"\' EXIT'
FAKE_SERVER = r'''
import http.server, os
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers(); self.wfile.write(b'ok')
    log_message = lambda *a: None
http.server.HTTPServer(('127.0.0.1', int(os.environ['PORT'])), H).serve_forever()
'''


class StopOnlyOurOwnServer(unittest.TestCase):
    """attrib_hq.sh must run stop_hq.sh from its EXIT trap only for a server it launched itself. Everything runs against a
    scratch HQ dir and a stub stop_hq.sh (ATTRIB_HQ / ATTRIB_SP); the real stop_hq.sh and HQ server are never involved."""

    def run_script(self, d, trap, foreign):
        d = Path(d)
        hq, sp, bin_ = d / 'hq', d / 'sp', d / 'bin'
        for x in (hq / 'single-user', sp, bin_):
            x.mkdir(parents=True)
        called, ownpid = d / 'stop-called', d / 'own.pid'
        (sp / 'stop_hq.sh').write_text(f'echo called >> {called}\nkill $(cat {ownpid} 2>/dev/null) 2>/dev/null\nexit 0\n')
        (bin_ / 'nvidia-smi').write_text('#!/bin/sh\necho 0\n')
        (bin_ / 'journalctl').write_text('#!/bin/sh\nexit 0\n')
        (d / 'server.py').write_text(FAKE_SERVER)
        # stands in for `nsys profile ... bash start_qwen.sh`; argv keeps "start_qwen.sh" in /proc/PID/cmdline
        (bin_ / 'nsys').write_text(f'#!/bin/sh\necho $$ > {ownpid}\nexec python3 {d}/server.py "$@"\n')
        for n in ('nvidia-smi', 'journalctl', 'nsys'):
            (bin_ / n).chmod(0o755)
        script = (PACK / 'gpu/attrib_hq.sh').read_text()
        self.assertEqual(script.count(NEW_TRAP), 1)
        script = script.replace(NEW_TRAP, trap)
        self.assertEqual(script.count('\ngates\ncpu_checks\n'), 1)
        script = script.replace('\ngates\ncpu_checks\n', '\ngates\n')           # cpu_checks needs the real venv and overlay
        (d / 'attrib_hq.sh').write_text(script)
        s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
        other = None
        if foreign:
            other = subprocess.Popen(['bash', '-c', 'exec -a start_qwen.sh sleep 120'])
            (hq / 'serve.pid').write_text(str(other.pid))
        try:
            r = subprocess.run(['bash', str(d / 'attrib_hq.sh')], capture_output=True, text=True, timeout=90, env={
                **os.environ, 'DRY_RUN': '0', 'CUDA_VISIBLE_DEVICES': '', 'ATTRIB_SP': str(sp), 'ATTRIB_HQ': str(hq),
                'ATTRIB_PORT': str(port), 'PATH': f'{bin_}:{os.environ["PATH"]}', 'OUT': str(d / 'out'),
                'XID_SINCE': time.strftime('%Y-%m-%d %H:%M:%S'), 'BUDGET_S': '40'})
            alive = other is not None and other.poll() is None
        finally:
            if other:
                other.kill(); other.wait()
            if ownpid.exists():
                try:
                    os.kill(int(ownpid.read_text()), 9)
                except (ProcessLookupError, ValueError):
                    pass
        return r, called.exists(), alive

    def test_gate_refusal_leaves_a_foreign_server_alone(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            r, called, alive = self.run_script(d, NEW_TRAP, foreign=True)
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertIn('still running', r.stdout + r.stderr)
        self.assertFalse(called, 'stop_hq.sh was run for a server this script did not launch')
        self.assertTrue(alive)

    def test_negative_control_old_trap_stops_the_foreign_server(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            r, called, alive = self.run_script(d, OLD_TRAP, foreign=True)
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertTrue(called)

    def test_failure_after_our_own_launch_stops_the_server(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            r, called, alive = self.run_script(d, NEW_TRAP, foreign=False)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('not on the V2 runner', r.stdout + r.stderr)
        self.assertTrue(called, 'stop_hq.sh must run for the server this script launched')


@unittest.skipUnless(GPUQ.exists() and (ITEMS / 'attrib-hq.toml').exists(), 'gpuq or items missing')
class GpuqClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(GPUQ))
        from gpuq import classify as C, manifest as M
        cls.C, cls.M = C, M
        cls.cfg = {'env': {'defaults': {'CUDA_HOME': 'x', 'CUDA_VISIBLE_DEVICES': '0,1'}, 'required': []}}

    def manifest(self, name):
        return self.M.parse(ITEMS / f'{name}.toml', self.cfg, check_paths=False)

    def classify(self, m, log, rc=0):
        return self.C.classify({'manifest': m, 'rc': rc, 'log': log, 'runtime': 1500, 'events': ''})[0]

    def dry_log(self, script, env):
        r = subprocess.run(['bash', str(PACK / 'gpu' / script)], capture_output=True, text=True, timeout=120,
                           env={**os.environ, 'DRY_RUN': '1', 'CUDA_VISIBLE_DEVICES': '', **env})
        return r.stdout + r.stderr

    def test_successful_logs_classify_ok_for_both_items(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            hq = self.dry_log('attrib_hq.sh', {'OUT': d})
            ll = self.dry_log('attrib_llama.sh', {'OUT': d})
        self.assertIn('kernel_log_faults=', hq)
        tail = ('[00:00:01 +1s] proof: \'Using [HOST_STAGED] all-reduce backends\'; host-staged lines 2; draft-head lines 2; block 16; k=5; KV pool 190000\n'
                '[00:09:00 +540s] driver rc=0\n[00:12:00 +720s] pack rc=0\n[00:12:00 +720s] done: /x/pack/budget.md\n'
                '{"name": "decode-8k", "kind": "decode", "ttft_s": 0.2, "stop_s": 7.1}\n[00:12:01 +721s] exit rc=0\n')
        for name, log in (('attrib-hq', hq + tail), ('attrib-llama', ll + tail)):
            self.assertEqual(self.classify(self.manifest(name), log), 'ok', name)

    def test_negative_control_old_log_key_and_old_signature_classified_hazard(self):
        m = self.manifest('attrib-hq')
        old = {**m, 'signatures': [{'class': 'hazard', 'regex': r'\bXid\b|fallen off the bus'}] + m['signatures']}
        line = '[00:00:01 +0s] gates: xid=0 gpu_max_used=12MiB ram_avail=9000MiB port_busy=no\n'
        self.assertEqual(self.classify(old, line), 'hazard')          # the defect: a clean gate line halts the queue
        self.assertEqual(self.classify(m, line), 'ok')                # no hazard signature in the new manifests
        for name in ('attrib-hq', 'attrib-llama'):
            self.assertFalse([s for s in self.manifest(name)['signatures'] if s['class'] == 'hazard'], name)

    def test_real_failure_codes_still_map(self):
        m = self.manifest('attrib-hq')
        self.assertEqual(self.classify(m, '', rc=3), 'hazard')
        self.assertEqual(self.classify(m, '', rc=8), 'insufficient_data')


if __name__ == '__main__':
    unittest.main()
