"""The liveness gate between tasks.

Run 8 lost its server at task 8 of 40. The loop wrote 33 rows with null token counts,
returned 0, and the driver tore down the box, so the crash reason was unrecoverable.
"""

from __future__ import annotations

import http.server
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.instruments import emit, run_polyglot


class Result:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.measures: dict[str, object] = {}


TASKS = [{"task_id": f"t{i}"} for i in range(5)]


class RunLegTest(unittest.TestCase):
    def leg(self, alive, metrics_url: str = "http://x/v1", jobs: int = 1):
        recorded: list[str] = []
        ran: list[str] = []

        def worker(task):
            ran.append(task["task_id"])
            return Result(task["task_id"])

        with mock.patch.object(run_polyglot, "server_alive", alive):
            rc = run_polyglot.run_leg(
                TASKS, worker, lambda r: recorded.append(r.task_id), metrics_url, jobs
            )
        return rc, ran, recorded

    def test_healthy_server_runs_every_task(self) -> None:
        rc, ran, recorded = self.leg(lambda url: True)
        self.assertEqual(rc, 0)
        self.assertEqual(recorded, [t["task_id"] for t in TASKS])

    def test_death_mid_leg_stops_instead_of_recording_failures(self) -> None:
        calls = {"n": 0}

        def alive(url: str) -> bool:
            calls["n"] += 1
            return calls["n"] <= 2

        rc, ran, recorded = self.leg(alive)
        self.assertEqual(rc, 5)
        # The two tasks that ran against a live server are kept; nothing after is invented.
        self.assertEqual(recorded, ["t0", "t1"])
        self.assertEqual(ran, ["t0", "t1"])

    def test_no_further_tasks_run_after_the_abort(self) -> None:
        # With jobs > 1 the remaining tasks are already queued; the event must stop them
        # calling the model rather than letting them each burn a reconnect timeout.
        calls = {"n": 0}
        lock = threading.Lock()

        def alive(url: str) -> bool:
            with lock:
                calls["n"] += 1
                return calls["n"] <= 1

        rc, ran, _ = self.leg(alive, jobs=4)
        self.assertEqual(rc, 5)
        self.assertLessEqual(len(ran), 1)

    def test_external_lane_has_no_endpoint_to_check(self) -> None:
        def alive(url: str) -> bool:
            raise AssertionError("must not probe when there is no local server")

        rc, _, recorded = self.leg(alive, metrics_url="")
        self.assertEqual(rc, 0)
        self.assertEqual(len(recorded), len(TASKS))


class ServerAliveTest(unittest.TestCase):
    """Against a real socket, so the URL shape and the failure path are both exercised."""

    class Handler(http.server.BaseHTTPRequestHandler):
        body = b'{"data": [{"id": "m"}]}'

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            self.wfile.write(self.body)

        def log_message(self, *args: object) -> None:
            pass

    def setUp(self) -> None:
        self.server = http.server.HTTPServer(("127.0.0.1", 0), self.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_answering_server_is_alive(self) -> None:
        self.assertTrue(emit.server_alive(self.url))

    def test_dead_port_is_not_alive(self) -> None:
        dead = http.server.HTTPServer(("127.0.0.1", 0), self.Handler)
        url = f"http://127.0.0.1:{dead.server_address[1]}/v1"
        dead.server_close()
        self.assertFalse(emit.server_alive(url, attempts=1, timeout=1.0))

    def test_a_single_blip_does_not_end_a_leg(self) -> None:
        # One dropped probe must not discard the remaining 30-odd tasks of a rented hour.
        calls = {"n": 0}
        real = emit.urllib.request.urlopen

        def flaky(url, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("connection reset")
            return real(url, timeout=timeout)

        with mock.patch.object(emit.urllib.request, "urlopen", flaky):
            self.assertTrue(emit.server_alive(self.url, attempts=3, timeout=2.0))


if __name__ == "__main__":
    unittest.main()
