"""Read-isolated task execution; the only network route is a local inference bridge."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import select
import socket
import socketserver
import subprocess
import threading


def relay(left, right):
    readers = [left, right]
    try:
        while readers:
            ready, _, _ = select.select(readers, [], [], 30)
            for source in ready:
                target = right if source is left else left
                data = source.recv(65536)
                if data:
                    target.sendall(data)
                else:
                    readers.remove(source)
                    target.shutdown(socket.SHUT_WR)
    except OSError:
        pass


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def inference_bridge(path, port):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            with socket.create_connection(("127.0.0.1", port), timeout=15) as upstream:
                upstream.settimeout(None)
                relay(self.request, upstream)

    server = ThreadedUnixServer(str(path), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def sandbox_command(workdir, assets, command, *, bridge=None, client_home=None, readonly=()):
    args = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/sbin", "/sbin", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--tmpfs", "/cargo", "--tmpfs", "/home",
            "--dir", "/home/bench", "--dir", "/etc", "--dir", "/runner",
            "--ro-bind", "/etc/alternatives", "/etc/alternatives",
            "--bind", str(Path(workdir).resolve()), "/work",
            "--ro-bind", assets["node_root"], "/node",
            "--ro-bind", assets["rust_root"], "/rust",
            "--ro-bind", assets["cargo_registry"], "/cargo/registry",
            "--ro-bind", assets["node_modules"], "/node_modules",
            "--ro-bind", assets["codex"], "/runner/codex",
            "--ro-bind", str(Path(__file__).resolve()), "/runner/isolation.py",
            "--setenv", "PATH", "/node/bin:/rust/bin:/usr/bin:/bin",
            "--setenv", "HOME", "/home/bench", "--setenv", "CARGO_HOME", "/cargo",
            "--setenv", "CARGO_NET_OFFLINE", "true", "--setenv", "CARGO_BUILD_JOBS", "2",
            "--setenv", "npm_config_offline", "true",
            "--setenv", "CARGO_TARGET_DIR", "/work/target", "--setenv", "CI", "1",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "TERM", "dumb",
            "--chdir", "/work"]
    for name in ("ld.so.cache", "passwd", "group", "hosts", "nsswitch.conf"):
        path = Path("/etc") / name
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]
    for relative in readonly:
        path = Path(workdir) / relative
        args += ["--ro-bind", str(path.resolve()), "/work/" + relative]
    if client_home is not None:
        args += ["--bind", str(Path(client_home).resolve()), "/codex-state",
                 "--setenv", "CODEX_HOME", "/codex-state"]
    if bridge is not None:
        args += ["--ro-bind", str(Path(bridge).resolve()), "/bridge",
                 "--", "/usr/bin/python3", "/runner/isolation.py", "inside", "--", *command]
    else:
        args += ["--", *command]
    return args


def run(workdir, assets, command, timeout, *, bridge=None, client_home=None, readonly=(),
        input_text=None, stdout_path=None, stderr_path=None):
    args = sandbox_command(workdir, assets, command, bridge=bridge,
                           client_home=client_home, readonly=readonly)
    with contextlib.ExitStack() as stack:
        stdout = stack.enter_context(Path(stdout_path).open("a")) if stdout_path else subprocess.PIPE
        stderr = stack.enter_context(Path(stderr_path).open("a")) if stderr_path else subprocess.PIPE
        process = subprocess.Popen(args, stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, text=True)
        try:
            out, err = process.communicate(input_text, timeout=timeout)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
    return subprocess.CompletedProcess(args, process.returncode, out or "", err or "")


def inside(command):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            with socket.socket(socket.AF_UNIX) as upstream:
                upstream.connect("/bridge")
                relay(self.request, upstream)

    server = ThreadedTCPServer(("127.0.0.1", 8081), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        return subprocess.call(command)
    finally:
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inside"])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return inside(command)


if __name__ == "__main__":
    raise SystemExit(main())
