#!/usr/bin/env python3
"""Parent-only frozen-binary CUDA checks. The default is a CPU-only selftest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import tarfile
import time
import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
MASTER = "869034b4b"
SCOPE = "4ab70fe9d7159352a8a1bbcdf26eeec247654c6e"
PARAMS = (r"hsk=256,hsv=256,nh=2,nr23=\[(1|2|4|8),1\],"
          r"kv=(512|2048|8192|32768|98304),nb=[1-8],mask=1,sinks=0,"
          r"max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,"
          r"type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,2,1,3\],kv_view=0")
NEW_CASES = (r"hsk=256,hsv=256,nh=4,nr23=\[(1|2|4|6|8),[12]\],"
             r"kv=(512|1024|4096),nb=[1-8],.*type_K=q8_0,type_V=q8_0")
EXPECTED_PERF = 320


def extract_source(tf, destination):
    destination = destination.resolve()
    for member in tf.getmembers():
        path = (destination / member.name).resolve()
        assert path == destination or destination in path.parents, "Archive path escapes worktree"
        assert member.isfile() or member.isdir() or member.issym() or member.islnk(), "Unexpected archive file type"
        if member.issym() or member.islnk():
            target = ((path.parent if member.issym() else destination) / member.linkname).resolve()
            assert target == destination or destination in target.parents, "Archive link escapes worktree"
    if hasattr(tarfile, "data_filter"):
        tf.extractall(destination, filter="data")
    else:
        tf.extractall(destination)


def perf_rows(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    rows = {}
    for case, value in re.findall(r"(FLASH_ATTN_EXT\([^\n]+?\)):\s+\d+ runs -\s+([0-9.]+) us/run", text):
        if case in rows:
            raise RuntimeError("Duplicate perf case: " + case)
        rows[case] = float(value)
    return rows


def summarize_perf(runs, expected):
    keys = set(runs[0][1])
    assert len(keys) == expected, (len(keys), expected)
    assert all(set(rows) == keys for _, rows in runs), "Incomplete/mismatched perf matrix"
    result = []
    for key in sorted(keys):
        up = statistics.mean(rows[key] for arm, rows in runs if arm == "up")
        scope = statistics.mean(rows[key] for arm, rows in runs if arm == "scope")
        loss = (scope / up - 1) * 100
        result.append({"case": key, "up_us_M": up, "scope_us_M": scope,
                       "latency_change_pct_D": loss, "regression_gt_2pct": loss > 2})
    return result


def identity(result):
    assert result.get("tokens"), "Empty generated tokens"
    assert not result.get("truncated", False), "Context truncated"
    assert result.get("timings", {}).get("predicted_per_second", 0) > 0, "Missing throughput"
    finish = tuple(result.get(key) for key in ("stop_type", "stopped_eos", "stopped_limit", "stopped_word", "stopping_word"))
    return result["tokens"], result["content"], finish


def selftest():
    sample = "  FLASH_ATTN_EXT(hsk=256):     3 runs -    100.00 us/run - 1 GFLOPS\n"
    assert perf_rows(sample) == {"FLASH_ATTN_EXT(hsk=256)": 100}
    rows = summarize_perf([("up", {"case": 100}), ("scope", {"case": 90}),
                           ("scope", {"case": 90}), ("up", {"case": 100})], 1)
    assert abs(rows[0]["latency_change_pct_D"] + 10) < 1e-9
    assert summarize_perf([("up", {"case": 100}), ("scope", {"case": 103})], 1)[0]["regression_gt_2pct"]
    for bad in ([('up', {})], [('up', {'case': 1}), ('scope', {})]):
        try:
            summarize_perf(bad, 1)
        except AssertionError:
            pass
        else:
            raise AssertionError("Incomplete-perf negative control passed")
    good = {"tokens": [1, 2], "content": "ok", "stop_type": "limit",
            "timings": {"predicted_per_second": 1}}
    assert identity(good) != identity(dict(good, tokens=[1, 3]))
    try:
        identity(dict(good, tokens=[]))
    except AssertionError:
        pass
    else:
        raise AssertionError("Empty-output negative control passed")
    probe = Run.__new__(Run)
    probe.args = SimpleNamespace(local=True, gpus="0,1", port=0)
    probe.gpu = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    probe.summary = {}
    probe.left = lambda: 30
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="NVRM: Xid 79", stderr="")):
        try:
            probe.xid()
        except AssertionError:
            pass
        else:
            raise AssertionError("Xid negative control passed")
    probe.xid = lambda: None
    probe.command = lambda *args, **kwargs: "12345\n"
    try:
        probe.guard()
    except AssertionError:
        pass
    else:
        raise AssertionError("Busy-GPU negative control passed")
    probe.args.local = False
    del probe.xid
    probe.command = lambda *args, **kwargs: "GPU Recovery Action : None\nGPU Recovery Action : Reset\n"
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="denied")):
        try:
            probe.xid()
        except AssertionError:
            pass
        else:
            raise AssertionError("Partial-recovery negative control passed")
    print("CPU selftest passed: perf parsing, completeness, >2% regression, output identity negative controls.")
    print("CPU selftest passed: Xid, busy-GPU and partial-recovery guards (mock inputs; no GPU tool executed).")
    print("Parent plan: frozen master/scope; eval and memcheck; up/scope/scope/up kernel perf; exact-PID servers.")
    print("D: 320 matched perf cases per boot. No GPU command was executed.")


class Run:
    def __init__(self, args):
        self.args = args
        assert 60 <= args.budget <= 3600
        self.start = time.monotonic()
        self.deadline = self.start + args.budget - 15
        self.out = Path(args.out or ROOT / "codex" / time.strftime("gpu-scope-%Y%m%d-%H%M%S")).resolve()
        assert ROOT in self.out.parents, "OUT must be inside this worktree"
        self.out.mkdir()
        self.cpu = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(self.out / "tmp"),
                        XDG_CACHE_HOME=str(self.out / "cache"), PYTHONDONTWRITEBYTECODE="1", GIT_NO_LAZY_FETCH="1",
                        GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="protocol.allow", GIT_CONFIG_VALUE_0="never")
        (self.out / "tmp").mkdir()
        self.gpu = dict(self.cpu, CUDA_VISIBLE_DEVICES=args.gpus, CUDA_CACHE_PATH=str(self.out / "cuda-cache"))
        self.server = None
        self.summary = {"arch": args.arch, "master": MASTER, "scope": SCOPE, "complete": False}

    def left(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise RuntimeError("INCOMPLETE: wall budget exhausted")
        return left

    def command(self, argv, log=None, gpu=False, env=None):
        cmd_env = env or (self.gpu if gpu else self.cpu)
        stream = (self.out / log).open("w") if log else None
        proc = subprocess.Popen(argv, env=cmd_env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=stream or subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        try:
            output, _ = proc.communicate(timeout=self.left())
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, argv, output)
        except BaseException:
            # This group contains only children of the captured command PID.
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
        finally:
            if stream:
                stream.close()
        return (self.out / log).read_text() if log else output

    def xid(self):
        journal = ["journalctl", "-k", "--no-pager"]
        journal += ["--since", "2026-10-01 13:29:00"] if self.args.local else ["-b"]
        proc = subprocess.run(journal,
                              env=self.gpu, capture_output=True, text=True, timeout=min(15, self.left()))
        if proc.returncode == 0 and proc.stdout.strip() and "No entries" not in proc.stdout:
            assert not re.search(r"Xid|fallen off", proc.stdout), "REFUSE: Xid/fallen off"
            self.summary["xid_guard"] = "kernel log"
            return
        if self.args.local:
            raise RuntimeError("REFUSE: kernel log unavailable; local run requires the dated Xid check")
        health = self.command(["nvidia-smi", "-i", self.args.gpus, "-q"], gpu=True)
        assert not re.search(r"Xid|fallen off|GPU is lost|Unknown Error", health, re.I), "REFUSE: GPU error"
        actions = re.findall(r"GPU Recovery Action\s*:\s*([^\r\n]+)", health)
        assert len(actions) == len(self.args.gpus.split(",")) and all(a.strip() == "None" for a in actions), "REFUSE: cannot establish GPU recovery state"
        assert not re.search(r"(?:Pending|Failure Occurred)\s*:\s*Yes", health), "REFUSE: pending GPU recovery"
        self.summary["xid_guard"] = "nvidia-smi recovery state (historical Xids unavailable)"

    def guard(self):
        self.xid()
        procs = self.command(["nvidia-smi", "-i", self.args.gpus, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], gpu=True)
        assert not procs.strip(), "REFUSE: GPU compute process is active"
        used = self.command(["nvidia-smi", "-i", self.args.gpus, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], gpu=True)
        assert all(int(x) <= 1500 for x in used.splitlines()), "REFUSE: GPUs hold memory"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.args.port))

    def git(self, *args):
        gitdir = ROOT / "codex/git"
        env = dict(self.cpu, GIT_NO_LAZY_FETCH="1", GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="protocol.allow", GIT_CONFIG_VALUE_0="never")
        if gitdir.exists():
            env.update(GIT_DIR=str(gitdir), GIT_WORK_TREE=str(ROOT))
        return self.command(["git", *args], env=env)

    def prepare(self):
        # An identical test matrix isolates production sources between the two arms.
        packaged = ROOT / "codex/source-inputs.json"
        inputs = json.loads(packaged.read_text()) if packaged.exists() else None
        if inputs:
            for arm, revision in (("up", MASTER), ("scope", SCOPE)):
                archive = ROOT / "codex" / inputs[arm]["archive"]
                assert inputs[arm]["revision"].startswith(revision), "Source revision mismatch"
                assert hashlib.sha256(archive.read_bytes()).hexdigest() == inputs[arm]["sha256"], "Source archive changed"
            with tarfile.open(ROOT / "codex" / inputs["scope"]["archive"]) as tf:
                tests = tf.extractfile("tests/test-backend-ops.cpp").read().decode()
        else:
            tests = self.git("show", f"{SCOPE}:tests/test-backend-ops.cpp")
        for arm, revision in (("up", MASTER), ("scope", SCOPE)):
            src = self.out / f"build-src-{arm}"
            src.mkdir()
            archive = ROOT / "codex" / inputs[arm]["archive"] if inputs else self.out / f"build-{arm}.tar"
            if not inputs:
                self.git("archive", "--format=tar", "-o", str(archive), revision)
            with tarfile.open(archive) as tf:
                extract_source(tf, src)
            (src / "tests/test-backend-ops.cpp").write_text(tests)
            build = self.out / f"build-{arm}"
            flags = ["-DGGML_CUDA=ON", "-DGGML_CUDA_NCCL=OFF", "-DGGML_CCACHE=OFF", "-DLLAMA_CURL=OFF",
                     "-DGGML_NATIVE=OFF", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CUDA_ARCHITECTURES={self.args.arch}",
                     "-DCMAKE_CUDA_FLAGS=--threads=1", "-DCUDAToolkit_ROOT=" + self.args.toolkit,
                     "-DCMAKE_CUDA_COMPILER=" + str(Path(self.args.toolkit) / "bin/nvcc")]
            self.command(["cmake", "-S", str(src), "-B", str(build), *flags], f"{arm}-config.log")
            targets = ["test-backend-ops", "llama-server"] if self.args.local or self.args.model else ["test-backend-ops"]
            self.command(["cmake", "--build", str(build), "--target", *targets, "-j8"], f"{arm}-build.log")
            frozen = self.out / "frozen" / arm
            frozen.mkdir(parents=True)
            for path in (build / "bin").glob("*.so*"):
                shutil.copy2(path, frozen / path.name)
            shutil.copy2(build / "bin/test-backend-ops", frozen / "test-backend-ops")
            if "llama-server" in targets:
                shutil.copy2(build / "bin/llama-server", frozen / "fa-scope-srv")
            cap = self.out / "cpu-thread-cap.cpp"
            cap.write_text("namespace std { struct thread { static unsigned int hardware_concurrency() noexcept; }; unsigned int thread::hardware_concurrency() noexcept { return 8; } }\n")
            self.command(["c++", "-shared", "-fPIC", str(cap), "-o", str(frozen / "cpu-thread-cap.so")])
            source = inputs[arm]["revision"] if inputs else self.git("rev-parse", revision).strip()
            (frozen / "SOURCE").write_text(json.dumps({"revision": source, "flags": flags}, indent=2) + "\n")
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir() if p.is_file()}
            (frozen / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2))
            print("Frozen", arm, revision, flush=True)

    def environment(self, arm, devices=None):
        frozen = self.out / "frozen" / arm
        hashes = json.loads((frozen / "SHA256SUMS.json").read_text())
        assert all(hashlib.sha256((frozen / p).read_bytes()).hexdigest() == h for p, h in hashes.items()), "Frozen binary changed"
        return dict(self.gpu, CUDA_VISIBLE_DEVICES=devices or self.args.gpus.split(',')[0],
                    LD_LIBRARY_PATH=str(frozen), LD_PRELOAD=str(frozen / "cpu-thread-cap.so"))

    def ops(self):
        for arm in ("up", "scope"):
            self.guard()
            binary = str(self.out / "frozen" / arm / "test-backend-ops")
            text = self.command([binary, "-b", "CUDA0", "-o", "FLASH_ATTN_EXT", "-j", "1"], f"{arm}-eval.log", env=self.environment(arm))
            assert "Backend CUDA0:" in text and "backends passed" in text and "FAIL" not in text, "CUDA eval incomplete"
            for nr in (1, 2, 4, 6, 8):
                for nb in range(1, 9):
                    assert re.search(rf"FLASH_ATTN_EXT\(hsk=256,hsv=256,nh=4,nr23=\[{nr},2\],kv=512,nb={nb},.*type_K=q8_0,type_V=q8_0.*OK", text), (nr, nb)
            self.xid()
        self.guard()
        binary = str(self.out / "frozen/scope/test-backend-ops")
        text = self.command(["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99", "--target-processes", "all",
                             binary, "-b", "CUDA0", "-o", "FLASH_ATTN_EXT", "-p", NEW_CASES, "-j", "1"],
                            "memcheck.log", env=self.environment("scope"))
        assert "ERROR SUMMARY: 0 errors" in text and "Backend CUDA0:" in text and "FAIL" not in text
        assert re.search(r"\d+/\d+ tests passed", text), "Memcheck selected no tests"
        self.xid()
        runs = []
        for boot, arm in enumerate(("up", "scope", "scope", "up"), 1):
            self.guard()
            binary = str(self.out / "frozen" / arm / "test-backend-ops")
            text = self.command([binary, "perf", "-b", "CUDA0", "-o", "FLASH_ATTN_EXT", "-p", PARAMS],
                                f"perf-{boot}-{arm}.log", env=self.environment(arm))
            rows = perf_rows(text)
            assert len(rows) == EXPECTED_PERF, (len(rows), EXPECTED_PERF)
            runs.append((arm, rows))
            self.xid()
        result = summarize_perf(runs, EXPECTED_PERF)
        (self.out / "perf.json").write_text(json.dumps(result, indent=2))
        self.summary["perf_regressions_gt_2pct"] = [r for r in result if r["regression_gt_2pct"]]
        self.summary["eval"] = "passed"
        self.summary["memcheck"] = "passed"

    def http(self, endpoint, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.args.port}/{endpoint}", data=data,
                                     headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=min(300, self.left())) as response:
            return json.load(response)

    def stop(self):
        if self.server is None:
            return
        if self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        self.server = None

    def boot(self, arm, devices, depth, tag):
        self.guard()
        model = self.args.model or str(Path.home() / "models/Qwen3.8-27B-IQ4_XS.gguf")
        assert Path(model).is_file(), "Public model missing; no download is performed"
        stat = Path(model).stat()
        assert (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) == self.model_stat, "Model input changed"
        flags = ["-m", model, "-np", "1", "-fa", "on", "-ngl", "99", "-t", "8", "-tb", "8", "-c", str(depth + 2048),
                 "-ctk", "q8_0", "-ctv", "q8_0", "-sm", "layer", "-b", "2048", "-ub", "512", "--cache-ram", "0",
                 "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
                 "--host", "127.0.0.1", "--port", str(self.args.port), "--metrics"]
        log = (self.out / f"{tag}-server.log").open("w")
        self.server = subprocess.Popen([str(self.out / "frozen" / arm / "fa-scope-srv"), *flags],
                                       env=self.environment(arm, devices), stdout=log, stderr=subprocess.STDOUT)
        log.close()
        health_deadline = min(self.deadline, time.monotonic() + 300)
        while time.monotonic() < health_deadline and self.server.poll() is None:
            try:
                if self.http("health").get("status") == "ok":
                    return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        self.stop()
        raise RuntimeError("Server boot failed: " + tag)

    def e2e(self):
        devices = self.args.gpus.split(',')[0]
        results = []
        for depth in (8192, 32768):
            for boot, arm in enumerate(("up", "scope", "scope", "up"), 1):
                tag = f"e2e-{depth}-{boot}-{arm}"
                try:
                    self.boot(arm, devices, depth, tag)
                except RuntimeError:
                    log = (self.out / f"{tag}-server.log").read_text()
                    assert self.args.local and ',' in self.args.gpus and re.search(r"out of memory|failed to allocate|unable to allocate", log, re.I), "Server failure other than GPU capacity"
                    devices = self.args.gpus
                    self.boot(arm, devices, depth, tag + "-two-gpu")
                tokens = self.http("tokenize", {"content": "A careful engineer measures the result before making a claim. ", "add_special": False})["tokens"]
                tail = self.http("tokenize", {"content": "\nContinue with a short explanation of matrix multiplication:\n", "add_special": False})["tokens"]
                prompt = (tokens * (depth // len(tokens) + 1))[:depth - len(tail)] + tail
                assert len(prompt) == depth
                for temp in (0, 0.8):
                    result = self.http("completion", {"prompt": prompt, "n_predict": 128, "temperature": temp, "seed": 12345,
                                                       "top_k": 20, "top_p": 0.95, "min_p": 0.05, "cache_prompt": False,
                                                       "return_tokens": True, "stream": False})
                    identity(result)
                    assert result["timings"].get("draft_n", 0) > 0, "MTP did not run; the public model needs an MTP head"
                    record = {"arm": arm, "depth": depth, "boot": boot, "temp": temp, "devices": devices, "result": result}
                    results.append(record)
                    (self.out / f"{tag}-temp{temp}.json").write_text(json.dumps(record, indent=2))
                self.stop()
                self.xid()
        for depth in (8192, 32768):
            for temp in (0, 0.8):
                group = [r for r in results if r["depth"] == depth and r["temp"] == temp]
                assert len({r["devices"] for r in group}) == 1, "Device layout changed between arms; rerun with --gpus set to the two-GPU layout"
                assert all(identity(r["result"]) == identity(group[0]["result"]) for r in group), "Output identity failed"
        self.summary["e2e"] = [{k: r[k] for k in ("arm", "depth", "boot", "temp", "devices")} |
                               {"tok_s_M": r["result"]["timings"]["predicted_per_second"]} for r in results]

    def execute(self):
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            self.guard()
            if self.args.local or self.args.model:
                model = Path(self.args.model or Path.home() / "models/Qwen3.8-27B-IQ4_XS.gguf")
                assert model.is_file(), "Public model missing; no download is performed"
                stat = model.stat()
                self.model_stat = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                digest = hashlib.sha256()
                with model.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        self.left()
                        digest.update(chunk)
                self.summary["model"] = {"path": str(model), "bytes_M": stat.st_size, "sha256": digest.hexdigest()}
            self.summary["hardware"] = self.command(["nvidia-smi", "-i", self.args.gpus, "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv"], gpu=True)
            for line in self.summary["hardware"].splitlines()[1:]:
                capability = line.split(',')[2].strip().replace('.', '')
                assert int(capability) == self.args.arch, "Requested architecture does not match the selected GPU"
            self.prepare()
            self.ops()
            if self.args.local or self.args.model:
                self.e2e()
            self.summary["complete"] = True
            self.summary["elapsed_s_M"] = time.monotonic() - self.start
            self.left()
            assert not self.summary["perf_regressions_gt_2pct"], "Kernel regressions >2%; see perf.json"
        except BaseException as error:
            self.summary["error"] = str(error)
            raise
        finally:
            self.stop()
            (self.out / f"summary-sm{self.args.arch}.json").write_text(json.dumps(self.summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--arch", type=int, choices=(75, 80, 86, 89, 120), default=120)
    parser.add_argument("--gpus", default=os.environ.get("SCOPE_GPUS", "0,1"))
    parser.add_argument("--budget", type=int, default=3600)
    parser.add_argument("--port", type=int, default=18097)
    parser.add_argument("--out")
    parser.add_argument("--model")
    parser.add_argument("--toolkit", default="/usr/local/cuda-13.2")
    args = parser.parse_args()
    if not args.run or args.selftest:
        selftest()
        return
    Run(args).execute()


if __name__ == "__main__":
    main()
