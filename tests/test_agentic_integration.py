import http.server
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from scripts.instruments import agentic_probe as probe
from scripts.instruments import isolated_exec


@unittest.skipUnless(os.environ.get("QWEN38_ISOLATION_TESTS") == "1", "requires host namespaces and prepared tool assets")
class IsolatedTrustedGraderTests(unittest.TestCase):
    def test_only_trusted_tests_count_and_starter_doctests_cannot_disappear(self):
        assets = json.loads(Path(".cache/agentic-assets-20260913/tools.json").read_text())
        with tempfile.TemporaryDirectory(prefix="qwen-grader-regression-") as temp:
            root = Path(temp)
            source = root / "sources/fixture"
            (source / "src").mkdir(parents=True)
            (source / "tests").mkdir()
            cargo = '[package]\nname="fixture"\nversion="0.1.0"\nedition="2021"\n'
            (source / "Cargo.toml").write_text(cargo)
            (source / "src/lib.rs").write_text(
                '/// ```compile_fail\n/// let _: fixture::Input = fixture::compute();\n/// ```\n'
                'pub struct Input;\npub struct Compute;\n'
                'pub fn compute() -> Compute { Compute }\npub fn answer() -> u8 { 42 }\n')
            trusted = '#[test]\n#[ignore]\nfn answer() { assert_eq!(fixture::answer(), 42); }\n'
            (source / "tests/main.rs").write_text(trusted)
            task = {"source_dir": "sources/fixture", "language": "rust", "test_paths": ["tests/main.rs"]}
            for name, code, expected in (
                ("valid", 'pub struct Input; pub struct Compute; pub fn compute() -> Compute { Compute }', True),
                ("bad_types", 'pub struct Input; pub type Compute = Input; pub fn compute() -> Compute { Input }', False),
            ):
                with self.subTest(name=name):
                    work = root / name
                    (work / "src").mkdir(parents=True)
                    (work / "tests").mkdir()
                    (work / "Cargo.toml").write_text(cargo)
                    (work / "tests/main.rs").write_text(trusted)
                    (work / "src/lib.rs").write_text(code + '\n'
                        '/// ```\n/// panic!("candidate doctest must not affect grading");\n/// ```\n'
                        'pub fn answer() -> u8 { 42 }\n'
                        '#[cfg(test)] mod own { #[test] fn fails() { panic!("not trusted"); } }\n')
                    grade = probe.test_suite(task, work, assets, 60, asset_dir=root)
                    self.assertEqual(grade["returncode"] == 0, expected, grade["output"])
                    self.assertEqual(grade["passed"], 2 if expected else 1, grade["output"])
                    self.assertEqual(grade["failed"], 0 if expected else 1, grade["output"])
                    self.assertEqual(grade["ignored"], 0)
                    self.assertTrue(grade["starter_doctests"])


@unittest.skipUnless(os.environ.get("QWEN38_ISOLATION_TESTS") == "1", "requires host namespaces and prepared tool assets")
class IsolatedClientTests(unittest.TestCase):
    def test_codex_tool_roundtrip_through_only_the_inference_bridge(self):
        assets = json.loads(Path(".cache/agentic-assets-20260913/tools.json").read_text())
        requests = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                if len(requests) == 1:
                    choices = []
                    for tool in request.get("tools", []):
                        if tool.get("type") == "namespace":
                            choices.extend((child, tool["name"]) for child in tool["tools"])
                        else:
                            choices.append((tool, None))
                    tool, namespace = next((tool, namespace) for tool, namespace in choices
                                           if tool.get("name") in ("exec_command", "shell"))
                    command = "test ! -e /home/xanewok/repos/qwen38-eval && test ! -e /work/.meta && echo ISOLATION_OK"
                    arguments = {"cmd": command} if tool["name"] == "exec_command" else {"command": ["sh", "-c", command]}
                    item = {"type": "function_call", "call_id": "isolation-test", "name": tool["name"],
                            "arguments": json.dumps(arguments)}
                    if namespace:
                        item["namespace"] = namespace
                else:
                    item = {"type": "message", "role": "assistant", "id": "done",
                            "content": [{"type": "output_text", "text": "Isolation check complete."}]}
                response_id = f"mock-{len(requests)}"
                events = [{"type": "response.created", "response": {"id": response_id}},
                          {"type": "response.output_item.done", "item": item},
                          {"type": "response.completed", "response": {"id": response_id,
                           "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}}}]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write("".join("data: " + json.dumps(event) + "\n\n" for event in events).encode())

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="qwen-isolation-test-") as temp:
                root = Path(temp)
                work, client_home = root / "work", root / "client"
                work.mkdir()
                client_home.mkdir()
                (client_home / "config.toml").write_text(probe.client_config(102400))
                with isolated_exec.inference_bridge(root / "bridge.sock", server.server_port):
                    result = isolated_exec.run(work, assets,
                        ["/runner/codex", "exec", "--skip-git-repo-check", "--json", "-"],
                        60, bridge=root / "bridge.sock", client_home=client_home,
                        input_text="Check the isolated workspace using a shell command.")
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertGreaterEqual(len(requests), 2, result.stderr + result.stdout)
                self.assertEqual(requests[0].get("reasoning", {}).get("effort"), "xhigh")
                self.assertNotEqual(requests[0].get("max_output_tokens"), 2048)
                outputs = [item for request in requests[1:] for item in request.get("input", [])
                           if item.get("type") == "function_call_output"]
                self.assertTrue(any("ISOLATION_OK" in str(item.get("output")) for item in outputs), outputs)
                denied = isolated_exec.run(work, assets, ["python3", "-c",
                    "import socket; s=socket.socket(); s.settimeout(.2); "
                    "assert s.connect_ex(('1.1.1.1',443)) != 0; print('external-network-denied')"], 10)
                self.assertEqual(denied.returncode, 0, denied.stderr)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
