"""gpu/attrib_llama_host.sh end to end on the CPU: a fake llama server (HTTP, SSE, draft counters), a fake nsys (launch / start /
stop / shutdown / export / sessions) that records every call and exports synthetic sqlite traces built from the real nsys
schema (tests/test_host_sampling.Builder), fake nvidia-smi / journalctl. The frozen stack, model and corpus checks are the
real ones (stat + sha only); nothing touches a GPU or the network.

What must hold: nsys start carries the CPU-sampling flags and nothing else replaces them; the server we launched is stopped
and a bystander is not; Xid and a busy port refuse before any launch; a trace without samples is rejected (exit 8) with
the raw files kept; the manifest maps the exit codes and does not turn a clean log into a hazard.
Run: CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_host_driver
"""
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path

from .test_host_sampling import Builder

PACK = Path(__file__).resolve().parent.parent
SP = Path(os.environ.get('ATTRIB_SCRATCH', '/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/'
                         '93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad'))
SCRIPT = PACK / 'gpu/attrib_llama_host.sh'
ITEM = PACK / 'gpu/items/attrib-llama-host.toml'
GPUQ = Path.home() / 'repos/gpuq'
STACK = Path('/home/xanewok/repos/qwen38-weekend/frozen/stack2/build/bin')
MODEL = Path.home() / 'models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf'

FAKE_SERVER = r'''#!/usr/bin/env python3
import http.server, json, os, sys, time
port = int(sys.argv[sys.argv.index('--port') + 1])
open(os.environ['FAKE_DIR'] + '/server.pid', 'w').write(str(os.getpid()))
open(os.environ['FAKE_DIR'] + '/server.argv', 'w').write(json.dumps(sys.argv[1:]))
cache = {}
K = 5

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    log_message = lambda *a: None

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}')

    def reply(self, obj):
        d = json.dumps(obj).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(d))); self.end_headers(); self.wfile.write(d)

    def do_GET(self):
        self.reply({'status': 'ok'})

    def do_POST(self):
        b = self.body()
        if self.path == '/tokenize':
            self.reply({'tokens': list(range(len(b['content'].split())))})
        elif self.path == '/detokenize':
            self.reply({'content': ' '.join(['w'] * len(b['tokens']))})
        elif self.path == '/apply-template':
            self.reply({'prompt': 'U: ' + b['messages'][0]['content'] + ' A:'})
        elif self.path == '/completion':
            self.completion(b)

    def completion(self, b):
        n = len(b['prompt'].split())
        hit = cache.get(b.get('id_slot', 0)) == b['prompt']
        cache[b.get('id_slot', 0)] = b['prompt']
        pn = 4 if hit else n
        base = {'prompt_n': pn, 'cache_n': n - pn, 'prompt_ms': pn * 0.4, 'prompt_per_second': 2500.0}
        t0 = time.monotonic()
        if not b.get('stream'):
            self.reply({'tokens_evaluated': n, 'timings': {**base, 'predicted_n': 1, 'predicted_ms': 1.0, 'draft_n': None}})
            return
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        N = b['n_predict']
        for i in range(N):
            self.wfile.write(b'data: ' + json.dumps({'content': 'x', 'tokens_predicted': i + 1}).encode() + b'\n\n')
            self.wfile.flush()
            time.sleep(0.003)
        rounds = N // 3
        t = {**base, 'predicted_n': N, 'predicted_ms': (time.monotonic() - t0) * 1e3, 'draft_n': rounds * K, 'draft_n_accepted': N - rounds}
        self.wfile.write(b'data: ' + json.dumps({'content': '', 'stop': True, 'timings': t}).encode() + b'\n\n')
        self.wfile.flush()

http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
'''

FAKE_NSYS = r'''#!/usr/bin/env python3
import json, os, shutil, signal, sys
D = os.environ['FAKE_DIR']
a = sys.argv[1:]
open(D + '/calls.log', 'a').write(json.dumps(a) + '\n')
cmd = a[0] if a else ''
sess = next((x.split('=', 1)[1] for x in a if x.startswith('--session=')), None)
os.makedirs(D + '/sessions', exist_ok=True)
if cmd == '--version':
    print('NVIDIA Nsight Systems version 2025.6.3 (fake)')
elif cmd == 'start' and '--help' in a:
    print('\t--samples-per-backtrace=\n\t--sampling-period=\n\t-b, --backtrace=')
elif cmd == 'launch':
    rest = [x for x in a[1:] if not x.startswith('--')]
    i = next(j for j, x in enumerate(a[1:], 1) if not x.startswith('--'))
    open(f'{D}/sessions/{sess}', 'w').write(str(os.getpid()))
    os.execvp(a[i], a[i:])
elif cmd == 'sessions':
    print('\n'.join(os.listdir(D + '/sessions')))
elif cmd == 'start':
    if os.environ.get('FAKE_NSYS_FAIL_START'):
        print('start failed', file=sys.stderr); sys.exit(1)
    if os.path.exists(f'{D}/capturing.{sess}'):
        print('already collecting', file=sys.stderr); sys.exit(1)
    open(f'{D}/capturing.{sess}', 'w').write(a[a.index('-o') + 1])
elif cmd == 'stop':
    try:
        out = open(f'{D}/capturing.{sess}').read()
    except OSError:
        print('not collecting', file=sys.stderr); sys.exit(1)
    open(out + '.nsys-rep', 'w').write('fake report')
    os.remove(f'{D}/capturing.{sess}')
elif cmd == 'shutdown':
    p = f'{D}/sessions/{sess}'
    if os.path.exists(p):
        if not os.environ.get('FAKE_NSYS_NOKILL'):
            try:
                os.kill(int(open(p).read()), signal.SIGTERM)
            except (OSError, ValueError):
                pass
        os.remove(p)
elif cmd == 'export':
    out, rep = a[a.index('-o') + 1], a[-1]
    name = os.path.basename(rep).split('.')[0]
    src = os.path.join(os.environ['FAKE_FIX'], name + '.sqlite')
    if not os.path.exists(src) or not os.path.exists(rep):
        print('export failed', file=sys.stderr); sys.exit(1)
    shutil.copy(src, out)
'''

needs = unittest.skipUnless(STACK.exists() and MODEL.exists(), 'frozen stack2 / model missing (script checks them for real)')


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Harness:
    def __init__(self, d, samples=True):
        d = Path(d)
        self.d, self.fake, self.fix, self.bin = d, d / 'fake', d / 'fix', d / 'bin'
        for x in (self.fake, self.fix, self.bin):
            x.mkdir()
        (self.fake / 'srv').write_text(FAKE_SERVER)       # never named llama-server: gpuq treats that name as a busy GPU
        (self.bin / 'nsys').write_text(FAKE_NSYS)
        (self.bin / 'nvidia-smi').write_text('#!/bin/sh\necho 0\necho 0\n')
        (self.bin / 'journalctl').write_text('#!/bin/sh\n[ -z "$FAKE_JOURNAL" ] || echo "$FAKE_JOURNAL"\nexit 0\n')
        for p in (self.fake / 'srv', self.bin / 'nsys', self.bin / 'nvidia-smi', self.bin / 'journalctl'):
            p.chmod(0o755)
        Builder(self.fix / 'decode-8k.sqlite')
        Builder(self.fix / 'decode-90k.sqlite', bare_every=3)
        if not samples:
            for f in self.fix.glob('*.sqlite'):
                db = sqlite3.connect(f)
                db.execute('drop table COMPOSITE_EVENTS')
                db.execute('drop table SAMPLING_CALLCHAINS')
                db.commit()
                db.close()
        self.out = d / 'out'

    def run(self, port=None, canary=False, timeout=240, **env):
        e = {**os.environ, 'DRY_RUN': '0', 'CUDA_VISIBLE_DEVICES': '', 'PATH': f'{self.bin}:{os.environ["PATH"]}', 'OUT': str(self.out),
             'FAKE_DIR': str(self.fake), 'FAKE_FIX': str(self.fix), 'SERVER_BIN': str(self.fake / 'srv'), 'PORT': str(port or free_port()),
             'XID_SINCE': '2999-01-01 00:00:00', 'BUDGET_S': '500', 'WINDOW_S': '0.6', 'DELAY_S': '0.3', 'N_PREDICT': '1500',
             'TWIN_N': '200', 'CANARY': '1' if canary else '0', 'N_PREDICT_CANARY': '1500', **env}
        return subprocess.run(['bash', str(SCRIPT)], capture_output=True, text=True, timeout=timeout, env=e)

    def calls(self):
        p = self.fake / 'calls.log'
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def server_alive(self):
        p = self.fake / 'server.pid'
        if not p.exists():
            return False
        try:
            os.kill(int(p.read_text()), 0)
            return True
        except ProcessLookupError:
            return False


@needs
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory(dir=SP)
        cls.h = Harness(cls.dir.name)
        cls.bystander = subprocess.Popen(['sleep', '300'])
        cls.r = cls.h.run()

    @classmethod
    def tearDownClass(cls):
        cls.bystander.kill()
        cls.bystander.wait()
        cls.dir.cleanup()

    def test_runs_to_the_host_report(self):
        r, out = self.r, self.h.out
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr + (out / 'console.log').read_text()[-2000:])
        for f in ('decode-8k.nsys-rep', 'decode-90k.nsys-rep', 'decode-8k.sqlite', 'runs.json', 'manifest.json', 'pack/budget.md',
                  'host/host.md', 'host/host.json'):
            self.assertTrue((out / f).exists(), f)
        self.assertFalse((out / 'prefill-32k.nsys-rep').exists())
        js = json.loads((out / 'host/host.json').read_text())
        self.assertEqual([t['depth'] for t in js['traces']], [8192, 90000])
        self.assertTrue(all(t['profiler_overhead'] for t in js['traces']), 'twin ms/round must reach the overhead line')
        self.assertIn('### Depth delta 8192 -> 90000', (out / 'host/host.md').read_text())

    def test_nsys_start_carries_the_sampling_flags_and_launch_keeps_the_cuda_trace(self):
        calls = self.h.calls()
        starts = [c for c in calls if c[0] == 'start' and '-o' in c]
        self.assertEqual(len(starts), 2)
        for c in starts:
            for flag in ('--sample=process-tree', '--backtrace=dwarf', '--samples-per-backtrace=1', '--sampling-period=2336904'):
                self.assertIn(flag, c)
            self.assertNotIn('--sample=none', c)
        self.assertEqual(len([c for c in calls if c[0] == 'stop']), 2)
        launch = next(c for c in calls if c[0] == 'launch')
        self.assertIn('--trace=cuda,nvtx,osrt', launch)
        self.assertIn('--cuda-graph-trace=node', launch)
        runs = json.loads((self.h.out / 'runs.json').read_text())
        self.assertIn('--backtrace=dwarf', runs['nsys_sample_args'])
        self.assertEqual([w['name'] for w in runs['windows']], ['decode-8k', 'decode-90k'])

    def test_server_flags_are_the_attrib_llama_ones(self):
        argv = json.loads((self.h.fake / 'server.argv').read_text())
        old = (PACK / 'gpu/attrib_llama.sh').read_text()
        for flag in ('--spec-type draft-mtp', '--spec-draft-n-max 5', '-sm tensor', '--cache-ram 0', '-b 2048', '-ub 128', '-np 1'):
            self.assertIn(flag, ' '.join(argv), flag)
        for line in re.findall(r'^  (--spec[^\n]*|--spec-draft-sampling[^\n]*)$', old, re.M):    # no flag drifted from attrib-llama
            for tok in line.split()[:1]:
                self.assertIn(tok, ' '.join(argv))

    def test_only_our_server_is_stopped(self):
        self.assertFalse(self.h.server_alive())
        self.assertIsNone(self.bystander.poll())
        shutdowns = [c for c in self.h.calls() if c[0] == 'shutdown']
        self.assertTrue(shutdowns and all(re.fullmatch(r'--session=attribhost\d+', c[1]) for c in shutdowns))

    def test_log_classifies_ok(self):
        if not (GPUQ.exists() and ITEM.exists()):
            self.skipTest('gpuq or item missing')
        sys.path.insert(0, str(GPUQ))
        from gpuq import classify as C, manifest as M
        cfg = {'env': {'defaults': {'CUDA_HOME': 'x', 'CUDA_VISIBLE_DEVICES': '0,1'}, 'required': []}}
        m = M.parse(ITEM, cfg, check_paths=False)
        log = (self.h.out / 'console.log').read_text() + self.r.stdout
        self.assertEqual(C.classify({'manifest': m, 'rc': 0, 'log': log, 'runtime': 400, 'events': ''})[0], 'ok')


@needs
class Refusals(unittest.TestCase):
    def test_trace_without_samples_is_rejected_and_raw_files_stay(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            h = Harness(d, samples=False)
            r = h.run()
            self.assertEqual(r.returncode, 8, r.stdout + r.stderr)
            md = (h.out / 'host/host.md').read_text()
            self.assertEqual(md.count('REJECTED'), 2)
            self.assertIn('no CPU samples', md)
            self.assertTrue((h.out / 'decode-90k.nsys-rep').exists() and (h.out / 'decode-8k.sqlite').exists())
            self.assertFalse(h.server_alive())

    def test_xid_in_the_kernel_log_refuses_before_any_launch(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            h = Harness(d)
            r = h.run(FAKE_JOURNAL='NVRM: Xid (PCI:0000:01:00): 13, Graphics Exception')
            self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
            self.assertFalse([c for c in h.calls() if c[0] == 'launch' or (c[0] == 'start' and '-o' in c)])

    def test_busy_port_refuses_and_leaves_the_foreign_listener_alone(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            h = Harness(d)
            port = free_port()
            foreign = subprocess.Popen([sys.executable, '-c', 'import http.server,sys;http.server.HTTPServer(("127.0.0.1",int(sys.argv[1])),'
                                        'http.server.BaseHTTPRequestHandler).serve_forever()', str(port)])
            try:
                for _ in range(50):
                    try:
                        socket.create_connection(('127.0.0.1', port), timeout=0.2).close()
                        break
                    except OSError:
                        time.sleep(0.1)
                r = h.run(port=port)
                self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
                self.assertFalse([c for c in h.calls() if c[0] == 'launch' or (c[0] == 'start' and '-o' in c)])
                self.assertIsNone(foreign.poll(), 'the script stopped a server it did not launch')
            finally:
                foreign.kill()
                foreign.wait()

    def test_nsys_start_failure_exits_2_and_stops_our_server(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            h = Harness(d)
            r = h.run(FAKE_NSYS_FAIL_START='1')
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertFalse(h.server_alive())

    def test_canary_is_8k_only_and_short(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            h = Harness(d)
            r = h.run(canary=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(len([c for c in h.calls() if c[0] == 'start' and '-o' in c]), 1)
            self.assertTrue((h.out / 'host/host.json').exists())
            self.assertFalse((h.out / 'decode-90k.nsys-rep').exists())

    def test_stray_files_are_moved_aside_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=SP) as d:
            h = Harness(d)
            h.out.mkdir()
            (h.out / 'decode-8k.nsys-rep').write_text('previous attempt')
            r = h.run(canary=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            prev = next(h.out.glob('prev-*'))
            self.assertEqual((prev / 'decode-8k.nsys-rep').read_text(), 'previous attempt')


class Drivers(unittest.TestCase):
    def test_default_nsys_start_keeps_the_cuda_only_flags(self):
        """llama_drive.py without --nsys-sample-args must behave exactly as for attrib-llama (--sample=none --cpuctxsw=none)."""
        sys.path.insert(0, str(PACK / 'gpu'))
        import llama_drive
        seen = []
        llama_drive.nsys = lambda *a, **k: (seen.append(a), type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})())[1]
        llama_drive.nsys_start('S', Path('/x/out'))
        self.assertEqual(seen[0], ('start', '--session=S', '--sample=none', '--cpuctxsw=none', '-o', '/x/out'))
        seen.clear()
        llama_drive.nsys_start('S', Path('/x/out'), ['--sample=process-tree', '--backtrace=dwarf'])
        self.assertEqual(seen[0], ('start', '--session=S', '--sample=process-tree', '--backtrace=dwarf', '-o', '/x/out'))


@unittest.skipUnless(GPUQ.exists() and ITEM.exists(), 'gpuq or item missing')
class Manifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(GPUQ))
        from gpuq import classify as C, manifest as M
        cls.C = C
        cls.cfg = {'env': {'defaults': {'CUDA_HOME': 'x', 'CUDA_VISIBLE_DEVICES': '0,1'}, 'required': []}}
        cls.m = M.parse(ITEM, cls.cfg, check_paths=False)
        cls.raw = tomllib.loads(ITEM.read_text())

    def classify(self, rc, log=''):
        return self.C.classify({'manifest': self.m, 'rc': rc, 'log': log, 'runtime': 400, 'events': ''})[0]

    def test_exit_codes(self):
        self.assertEqual({k: self.classify(int(k)) for k in ('3', '4', '5', '6', '7', '8')},
                         {'3': 'hazard', '4': 'transient', '5': 'insufficient_data', '6': 'integrity', '7': 'timeout',
                          '8': 'insufficient_data'})
        self.assertEqual(self.classify(2), 'deterministic')
        self.assertEqual(self.classify(0, 'gates ok: GPUs 123 MiB, port 18100 free\n'), 'ok')

    def test_signatures_are_word_bounded_and_there_is_no_hazard_text_rule(self):
        for s in self.m['signatures']:
            self.assertNotEqual(s['class'], 'hazard', 'a harmless log line would halt the queue')
        self.assertEqual(self.classify(0, 'stop_xid_hash_s_G fallen off\n'), 'ok')      # graph-reuse-tandem 10-03: substring Xid halted the queue

    def test_script_uses_exit_3_and_7_only_for_xid_and_budget(self):
        """gpuq turns an unmapped rc 3/7 into a queue-halting hazard: 3 is reserved for the kernel-log Xid, 7 is the budget."""
        text = SCRIPT.read_text()
        self.assertLessEqual(set(re.findall(r'(?:exit|return) ([0-9]+)', text)), {'0', '1', '2', '3', '4', '6', '7', '8', '130'})      # 1 = alive() predicate
        for line in re.findall(r'.*(?:exit|return) 3\b.*', text):
            self.assertIn('Xid', line)
        for line in re.findall(r'.*exit 7\b.*', text):
            self.assertIn('budget', line)

    def test_pins_cover_every_file_the_run_reads(self):
        pins = set(self.raw['pins']['files'])
        for f in ('gpu/attrib_llama_host.sh', 'gpu/llama_drive.py', 'gpu/make_manifest.py', 'gpu/inputs.sha256', 'gpu/inputs/agg.py',
                  'gpu/inputs/corpus.txt', 'gpu/inputs/chat-template.jinja', 'attrib_pack.py', 'host_sampling.py',
                  'models/llama-swift15-iq4xs.json'):
            self.assertIn(str(PACK / f), pins, f)
        self.assertIn(str(STACK / 'weekend-srv'), pins)
        self.assertIn(str(STACK / 'SHA256SUMS'), pins)

    def test_canary_is_8k_only_and_inside_the_canary_cap(self):
        c = self.raw['canary']
        self.assertEqual(c['env']['CANARY'], '1')
        self.assertLessEqual(c['timeout_s'], 900)
        self.assertEqual(self.raw['run']['env']['CANARY'], '0')
        self.assertLessEqual(self.raw['est_s'], self.raw['budget_s'])


if __name__ == '__main__':
    unittest.main()
