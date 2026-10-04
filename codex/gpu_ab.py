#!/usr/bin/env python3
"""Frozen mask A/B. Only the parent may select --execute-parent."""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request

HAZARD = re.compile(r"\bXid\b|\bfallen off\b", re.I)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen(root):
    source = (root / "SOURCE").read_text().splitlines()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("invalid frozen SOURCE")
    files = {}
    for directory in (root, root / "bin"):
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(None, 1)
            if name.startswith("*"):
                name = name[1:]
            path = directory / name
            if not path.resolve().is_relative_to(root.resolve()) or sha(path) != digest:
                raise ValueError("frozen bytes changed: " + name)
            files[str(path)] = digest
    binary = root / "bin/host-depth-server"
    if str(binary) not in files or str(root / "SOURCE") not in files or str(root / "bin/SHA256SUMS") not in files:
        raise ValueError("incomplete frozen manifests")
    if {p.name for p in (root / "bin").iterdir()} != {Path(p).name for p in files if Path(p).parent == root / "bin"}:
        raise ValueError("unhashed binary files")
    return binary, files


def loaded_outside(pid, root):
    libs = {line.split()[-1] for line in Path(f"/proc/{pid}/maps").read_text().splitlines()
            if re.search(r"/lib(llama|ggml|mtmd)[^/]*\.so", line)}
    return sorted(p for p in libs if not Path(p).resolve().is_relative_to(root))


def gpu_memory(raw, devices):
    rows = list(csv.DictReader(io.StringIO(raw)))
    found = {}
    for row in rows:
        row = {k.strip(): v.strip() for k, v in row.items()}
        found[row["uuid"]] = int(row["memory.used [MiB]"].split()[0])
    if not set(devices) <= found.keys():
        raise ValueError("selected GPU missing")
    return sum(found[d] for d in devices)


def analyze(record, minimum=40):
    chunks = record["chunks"]
    valid = [c["timings"] for c in chunks if isinstance(c.get("timings"), dict) and
             all(isinstance(c["timings"].get(k), (int, float)) for k in ("predicted_ms", "predicted_n", "draft_n", "draft_n_accepted"))]
    if not valid:
        raise ValueError("missing final timing counters")
    final = valid[-1]
    if any(not math.isfinite(v) or v < 0 for v in final.values() if isinstance(v, (int, float))):
        raise ValueError("invalid timing")
    if final["draft_n"] != int(final["draft_n"]) or not 0 <= final["draft_n_accepted"] <= final["draft_n"]:
        raise ValueError("invalid draft counters")
    points = {}
    last_n = last_ms = 0
    for t in valid:
        n, ms = t["draft_n"], t["predicted_ms"]
        if n < last_n or ms < last_ms:
            raise ValueError("reversed cumulative timing")
        last_n, last_ms = n, ms
        if n > 0 and n % 5 == 0 and t["predicted_n"] <= final["predicted_n"] - 6:
            points[n//5] = ms
    keys = sorted(points)
    intervals = [points[b] - points[a] for a, b in zip(keys, keys[1:]) if a >= 3 and b == a + 1 and points[b] > points[a]]
    if len(intervals) < minimum:
        raise ValueError(f"insufficient full-K interior intervals: {len(intervals)} < {minimum}")
    return {"intervals_M": len(intervals), "steady_ms_per_round_D": statistics.mean(intervals), "timings": final}


def identity(record):
    chunks = record["chunks"]
    final = next((c for c in reversed(chunks) if c.get("stop") is True), None)
    # A streamed final chunk carries empty tokens/content (server-context.cpp send_final_response); only the
    # partial chunks hold them, so concatenate. A non-streamed reply is a single final chunk with all tokens.
    tokens = [t for c in chunks for t in (c.get("tokens") or [])]
    if final is None or not tokens:
        raise ValueError("missing final token identity")
    return tokens, {k: final["timings"].get(k) for k in ("predicted_n", "draft_n", "draft_n_accepted", "prompt_n", "cache_n")}


def fixture_test(directory):
    record = json.loads((directory / "llama-sse/long-tool-s1-1.json").read_text())
    result = analyze(record, 1)
    changed = copy.deepcopy(record)
    changed["chunks"].extend([{}, {"timings": None}, {"timings": {"draft_n": None}}, record["chunks"][-1]])
    assert analyze(changed, 1) == result
    try:
        analyze(record, 40)
    except ValueError as e:
        assert "insufficient" in str(e)
    else:
        raise AssertionError("real short fixture escaped size gate")
    devices = ["GPU-ee9a3956-297c-604b-b4c1-f07930d27e46", "GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f"]
    assert gpu_memory((directory / "nvidia-smi-query.csv").read_text(), devices) == 123
    assert HAZARD.search((directory / "xid-lines.txt").read_text())
    assert not HAZARD.search("stop_xid_hash_s_G")
    # Recorded chat output has no native token IDs. The live canary must exercise that gate.
    try:
        identity(record)
    except ValueError:
        pass
    else:
        raise AssertionError("identity accepted a record without native token IDs")
    print(json.dumps({"recorded_fixture": str(directory), "parser_result": result, "negative_controls": "PASS"}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", type=Path)
    p.add_argument("--up", type=Path); p.add_argument("--patch", type=Path)
    p.add_argument("--model", type=Path); p.add_argument("--model-sha")
    p.add_argument("--devices", default="GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f")
    p.add_argument("--out", type=Path, default=Path("codex/gpu-runs"))
    p.add_argument("--port", type=int, default=int(os.environ.get("GPUQ_PORT", "8096")))
    p.add_argument("--draft-vocab", type=Path,
                   default=Path("/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"))
    p.add_argument("--draft-vocab-sha", default="e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6")
    p.add_argument("--budget", type=int, default=3600)
    p.add_argument("--canary", action="store_true")
    p.add_argument("--execute-parent", action="store_true")
    args = p.parse_args()
    if args.self_test:
        fixture_test(args.self_test); return
    if any(x is None for x in (args.up, args.patch, args.model, args.model_sha)):
        p.error("provide frozen roots, model and model SHA")
    roots = {"up": args.up.resolve(), "patch": args.patch.resolve()}
    binaries, pins = {}, {}
    digests = {}
    for arm, root in roots.items():
        binaries[arm], files = frozen(root); pins.update(files)
        digests[arm] = sorted(files[k] for k in files if Path(k).parent == root / "bin")
    if (roots["up"] / "SOURCE").read_text() == (roots["patch"] / "SOURCE").read_text() or digests["up"] == digests["patch"]:
        raise ValueError("frozen bytes changed: up and patch arms are not distinct builds")
    if sha(args.draft_vocab) != args.draft_vocab_sha:
        raise ValueError("draft vocab changed")
    pins[str(args.draft_vocab.resolve())] = args.draft_vocab_sha
    if sha(args.model) != args.model_sha:
        raise ValueError("model bytes changed")
    pins[str(args.model.resolve())] = args.model_sha
    for script in (Path(__file__), Path(__file__).with_name("gpu_ab.sh")):
        pins[str(script.resolve())] = sha(script)
    order = ["patch"] if args.canary else ["up", "patch", "patch", "up"]
    budget = min(args.budget, 600) if args.canary else args.budget
    estimate = 400 if args.canary else 480 * len(order)
    if estimate > .7 * budget:
        raise ValueError("stage estimate exceeds 70 percent of wall budget")
    plan = {"order": order, "depths": [8192, 90112], "k": 5, "pins": pins,
            "stage_estimate_s_G": estimate, "budget_s": budget, "steady_minimum": 40}
    if not args.execute_parent:
        print(json.dumps(plan, indent=2)); return
    if os.environ.get("GPUQ_ACTIVE") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("parent execution requires GPUQ_ACTIVE=1 and visible GPUs")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "pins.json").write_text(json.dumps(plan, indent=2))
    start = time.monotonic(); cutoff = str(time.time())
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text()
    devices = args.devices.split(",")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    proc = None
    results = {}

    def left():
        value = budget - (time.monotonic() - start)
        if value <= 0:
            raise RuntimeError("budget expired")
        return value

    def guard():
        left()
        if Path("/proc/sys/kernel/random/boot_id").read_text() != boot_id:
            raise RuntimeError("boot changed")
        raw = subprocess.check_output(["journalctl", "-k", "-b", "--since", "@" + cutoff, "--no-pager"], text=True, timeout=min(10, left()))
        if HAZARD.search(raw):
            raise RuntimeError("hazard: kernel fault since cutoff")

    def http(path, body=None):
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}" + path,
            data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=min(3, left()) if path == "/health" else left())

    def stop():
        nonlocal proc
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=min(30, max(1, left())))
            except (subprocess.TimeoutExpired, RuntimeError):
                proc.kill(); proc.wait(timeout=10)
        proc = None

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("terminated")))
    try:
        for i, arm in enumerate(order):
            guard()
            while True:
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv"], text=True, timeout=min(10, left()))
                if gpu_memory(raw, devices) <= 1500:
                    break
                time.sleep(min(2, left())); guard()
            while int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) < 6*1024*1024:
                time.sleep(min(2, left())); guard()
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", args.port))
                except OSError as e:
                    raise RuntimeError(f"port busy: {e}")
            binary, _ = frozen(roots[arm])
            dest = args.out / f"{i}-{arm}"; dest.mkdir()
            command = [str(binary), "-m", str(args.model.resolve()), "-c", "102400", "-np", "1", "-fa", "on",
                "-ctk", "q8_0", "-ctv", "q8_0", "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16",
                "-sm", "tensor", "--tensor-split", "50,50", "-ngl", "99", "-b", "2048", "-ub", "128",
                "--spec-type", "draft-mtp", "--spec-draft-n-max", "5", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
                "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling",
                "--spec-draft-vocab", str(args.draft_vocab.resolve()), "--cache-ram", "0", "--metrics", "-t", "8", "-tb", "8",
                "--host", "127.0.0.1", "--port", str(args.port)]
            with (dest / "server.log").open("w") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=log, env=dict(env, LD_LIBRARY_PATH=str(binary.parent)))
                (dest / "pid").write_text(str(proc.pid))
                (dest / "command.json").write_text(json.dumps(command))
                deadline = time.monotonic() + min(180, left())
                while True:
                    if proc.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("server failed readiness")
                    try:
                        with http("/health") as reply:
                            if json.load(reply).get("status") == "ok":
                                outside = loaded_outside(proc.pid, roots[arm])
                                if outside:
                                    raise ValueError("frozen bytes changed: loaded outside frozen root: " + ", ".join(outside))
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    time.sleep(min(1, left())); guard()
                records, failures = {}, []
                for depth in (8192, 90112):
                    body = {"prompt": [1000] * depth, "n_predict": 512, "ignore_eos": True, "cache_prompt": False,
                            "seed": 1729, "temperature": 1, "top_k": 20, "top_p": .95, "min_p": .05,
                            "stream": True, "timings_per_token": True, "return_tokens": True}
                    record = {"request": body, "chunks": []}
                    path = dest / f"depth-{depth}.json"
                    try:
                        with http("/completion", body) as reply:
                            for line in reply:
                                left()
                                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                                    record["chunks"].append(json.loads(line[6:]))
                    except Exception as e:
                        record["error"] = "budget expired" if time.monotonic() - start >= budget else str(e)
                        if isinstance(e, urllib.error.HTTPError):
                            record["error_body"] = e.read().decode(errors="replace")
                    finally:
                        path.write_text(json.dumps(record))
                    records[depth] = record
                    guard()
                for depth, record in records.items():
                    try:
                        if "error" in record:
                            raise ValueError(record["error"])
                        result = analyze(record)
                        ident = identity(record)
                        if result["timings"]["predicted_n"] < 256 or result["timings"].get("cache_n") != 0 or result["timings"].get("prompt_n") != depth:
                            raise ValueError("length or prefill gate failed")
                        if len(ident[0]) != result["timings"]["predicted_n"]:
                            raise ValueError("token count gate failed")
                        if depth in results and results[depth]["identity"] != ident:
                            raise ValueError("token or acceptance identity failed")
                        if args.canary and depth == 8192:
                            # A second identical request in the same boot tests RNG reset and the identity gate.
                            repeated = {"chunks": []}
                            try:
                                with http("/completion", dict(record["request"], stream=False)) as reply:
                                    repeated["chunks"].append(json.load(reply))
                            except Exception as e:
                                repeated["error"] = str(e)
                                if isinstance(e, urllib.error.HTTPError):
                                    repeated["error_body"] = e.read().decode(errors="replace")
                            finally:
                                (dest / f"repeat-{depth}.json").write_text(json.dumps(repeated))
                            guard()
                            if identity(repeated) != ident:
                                raise ValueError("repeat identity failed")
                        results.setdefault(depth, {"identity": ident, "runs": []})["runs"].append({"arm": arm, **result})
                    except Exception as e:
                        failures.append(f"depth {depth}: {e}")
                (dest / "verdict.json").write_text(json.dumps({"failures": failures}, indent=2))
                stop(); guard()
                if failures:
                    raise RuntimeError("INVALID: " + "; ".join(failures))
        if not args.canary:
            for result in results.values():
                up = [r["steady_ms_per_round_D"] for r in result["runs"] if r["arm"] == "up"]
                patch = [r["steady_ms_per_round_D"] for r in result["runs"] if r["arm"] == "patch"]
                result["patch_minus_up_ms_D"] = statistics.mean(patch) - statistics.mean(up)
        (args.out / "summary.json").write_text(json.dumps(results, indent=2))
    except Exception as e:
        reason = str(e)
        verdict = "hazard" if "hazard:" in reason else "budget" if "budget" in reason else "INVALID"
        (args.out / "verdict.json").write_text(json.dumps({"verdict": verdict, "reason": reason}, indent=2))
        raise
    finally:
        stop()


def exit_code(reason):
    # gpuq: 3/7 halt the queue as hazards; the daemon detects Xid itself, so never return them from here.
    if "port busy" in reason:
        return 4
    if re.search(r"frozen bytes changed|model bytes changed|draft vocab changed|identity failed", reason):
        return 6
    if "insufficient" in reason:
        return 8
    return 1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{type(e).__name__}: {e}", flush=True)
        raise SystemExit(exit_code(str(e)))
