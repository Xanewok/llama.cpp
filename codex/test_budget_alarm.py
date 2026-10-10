"""CPU test for the window runners' boot-wall alarm (gpu_stack7/8/9.py beside this file).

Contract: the alarm raises a BaseException, never a ValueError. http.client turns a ValueError raised while it waits for a
chunk size into IncompleteRead(b''), so lib.Fail from the handler read as a cut SSE stream (vanilla np2 90k, 10-10).
Run: python3 -P codex/test_budget_alarm.py   (HARNESS_CODE_ROOT defaults to the main checkout)
"""
import importlib.util
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import unittest
from unittest import mock

EVAL = Path(os.environ.get('S7_EVAL', '/home/xanewok/repos/qwen38-eval'))
os.environ.setdefault('HARNESS_CODE_ROOT', str(EVAL))
sys.path.insert(0, os.environ['HARNESS_CODE_ROOT'])
from harness import lib, depth, campaign

CODEX = Path(__file__).resolve().parent
RUNNERS = sorted(CODEX.glob('gpu_stack[789].py'))


def load(path):
    spec = importlib.util.spec_from_file_location('runner_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gapped_sse_url():
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n')
        time.sleep(0.05)
        conn.sendall(b'3\r\n:\n\n\r\n')
        time.sleep(3)
        conn.close()
        srv.close()
    threading.Thread(target=serve, daemon=True).start()
    return f'http://127.0.0.1:{srv.getsockname()[1]}'


def stream_with_alarm(handler):
    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        return depth.stream_depth(gapped_sse_url(), {}, 10, lambda: None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def depth_experiment():
    for path in sorted((EVAL / 'harness/experiments').glob('*.toml')):
        try:
            x = lib.read_experiment(path)
        except Exception:
            continue
        if x['kind'] == 'depth' and x['depth'].get('wall_s', 0) > 120:
            return path
    return None


class BudgetAlarm(unittest.TestCase):
    def test_a_valueerror_alarm_masquerades_as_incompleteread(self):
        """Negative control: the stream client swallows a ValueError alarm, so the checks below can fail."""
        def swallowed(signum, frame):
            raise ValueError('budget: boot wall expired')
        record = stream_with_alarm(swallowed)
        self.assertEqual(record['error'], 'IncompleteRead: IncompleteRead(0 bytes read)')

    def test_lib_fail_is_the_valueerror_the_old_handler_raised(self):
        self.assertTrue(issubclass(lib.Fail, ValueError))

    def test_runner_alarm_survives_the_stream_client_and_maps_to_rc7(self):
        self.assertTrue(RUNNERS)
        for path in RUNNERS:
            with self.subTest(runner=path.name):
                m = load(path)
                self.assertFalse(issubclass(m.BudgetExpired, Exception))
                with self.assertRaises(m.BudgetExpired):
                    stream_with_alarm(m.expire_budget)

                def expiring():
                    signal.signal(signal.SIGALRM, m.expire_budget)
                    signal.setitimer(signal.ITIMER_REAL, 60)
                    raise m.BudgetExpired('unit')
                previous = signal.getsignal(signal.SIGALRM)
                try:
                    self.assertEqual(m.guarded(expiring), 7)
                    self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
                    self.assertEqual(m.guarded(lambda: 3), 3)
                    with self.assertRaises(RuntimeError):
                        m.guarded(mock.Mock(side_effect=RuntimeError('harness bug')))
                finally:
                    signal.signal(signal.SIGALRM, previous)

    def test_server_start_arms_the_runner_alarm_and_stop_disarms_it(self):
        experiment = depth_experiment()
        if experiment is None:
            self.skipTest('no depth experiment in harness/experiments')
        for path in RUNNERS:
            with self.subTest(runner=path.name):
                m = load(path)
                previous = signal.getsignal(signal.SIGALRM)
                try:
                    with mock.patch.object(lib.Server, 'start', lambda self: 1), \
                         mock.patch.object(lib.Server, 'stop', lambda self, grace=30: None), \
                         mock.patch.object(lib, 'server_env', lib.server_env), \
                         mock.patch.object(lib, 'record_journal', lib.record_journal), \
                         mock.patch.object(campaign, 'harness_cmd', campaign.harness_cmd):
                        m.install_runtime_guards(['run', str(experiment)])
                        lib.Server.start(object())
                        self.assertIs(signal.getsignal(signal.SIGALRM), m.expire_budget)
                        self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
                        lib.Server.stop(object())
                        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    unittest.main()
