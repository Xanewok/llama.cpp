#!/usr/bin/env python3
import argparse
import collections
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "frozen/gdn-getter-stack6"
BIN = ROOT / "build/bin"
OPS = {"GATED_DELTA_NET", "GATED_DELTA_NET_CACHE_FUSION", "GATED_DELTA_NET_INDEXED", "GATED_DELTA_NET_INDEXED_CACHE_FUSION"}
SANITIZER = "/usr/local/cuda-13.2/bin/compute-sanitizer"
RECORDED = HERE / "fixtures"
FIXTURES = Path("/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures")

ANSI = re.compile(r'\x1b\[[0-9;]*m')
CASE = re.compile(r'(?:^|\s)([A-Z][A-Z0-9_]+)\((.*?)\): .*?(OK|FAIL|not supported.*)$')
SUMMARY = re.compile(r'(\d+)/(\d+) tests passed')
ERR_SUMMARY = re.compile(r'ERROR SUMMARY: (\d+) errors?')
RACE_SUMMARY = re.compile(r'RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)')
XID = re.compile(r'NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+), pid=([^,]*), name=([^,]*),')
SAN_NOT_ERROR = ('COMPUTE-SANITIZER', 'ERROR SUMMARY', 'RACECHECK SUMMARY', 'Target application returned an error')
GDN_LINE = re.compile(r'gdn_state_indexed: device=0 graph=\d+ tokens=(\d+) gathers_skipped=(\d+)')

def parse_cases(text, op=None):
    c = collections.Counter()
    for ln in text.splitlines():
        m = CASE.search(ANSI.sub('', ln).rstrip())
        if m and op in (None, m.group(1)):
            c['NS' if m.group(3).startswith('not') else m.group(3)] += 1
    return dict(ok=c['OK'], fail=c['FAIL'], ns=c['NS'])


def parse_gdn_cases(text):
    ops = {o: parse_cases(text, o) for o in OPS}
    return dict(ops=ops, cases={k: sum(c[k] for c in ops.values()) for k in ('ok', 'fail', 'ns')})


def parse_summary(text):
    m = SUMMARY.findall(text)
    return f'{m[-1][0]}/{m[-1][1]}' if m else None


def parse_sanitizer(text):
    san = [ln for ln in text.splitlines() if ln.startswith('=========')]
    headers = [i for i, ln in enumerate(san)
               if re.match(r'========= [^ ]', ln) and not any(k in ln for k in SAN_NOT_ERROR)]
    e, r = ERR_SUMMARY.findall(text), RACE_SUMMARY.findall(text)
    first = headers[0] if headers else None
    return dict(banner=any('COMPUTE-SANITIZER' in ln for ln in san),
                errors=int(e[-1]) if e else None,
                hazards=int(r[-1][1]) if r else None,  # racecheck errors only: `all` also displays benign analysis hazards
                headers=len(headers),
                first_lines=san[first:first + 50] if first is not None else [])


def parse_engagement(text):
    g = [(int(m.group(1)), int(m.group(2))) for m in GDN_LINE.finditer(text)]
    return dict(gdn_indexed_lines=len(g), tokens=sorted({t for t, _ in g}), all_skipped=bool(g) and all(k >= 1 for _, k in g))


def parse_xids(text):
    return [dict(code=int(m.group(2)), pid=m.group(3), name=m.group(4), line=ln.strip())
            for ln in text.splitlines() for m in [XID.search(ln)] if m]



def judge(tool, target, sanitizer, expected, rc):
    cases = parse_gdn_cases(target)
    san = parse_sanitizer(sanitizer)
    errors = []
    if rc != 0 or cases["cases"]["fail"] or san["headers"] or (san["errors"] or 0) or (san["hazards"] or 0):
        errors.append("finding")
    if cases["cases"]["ok"] != expected or cases["cases"]["ns"]:
        errors.append("case coverage")
    if not san["banner"] or (san["hazards"] if tool == "racecheck" else san["errors"]) is None:
        errors.append("sanitizer summary missing")
    if parse_summary(target) is None or "RTX 5060 Ti" not in target:
        errors.append("backend summary missing")
    if "GATED_DELTA_NET_INDEXED(" in target:
        engaged = parse_engagement(target)
        if not engaged["all_skipped"]:
            errors.append("path-engaged proof failed")
    return errors


def selftest():
    for name, tool, expected in [("mc-old", "memcheck", 41), ("mc-idx", "memcheck", 5),
                                 ("rc-idx", "racecheck", 5), ("sc-idx", "synccheck", 5)]:
        text = (RECORDED / (name + ".log")).read_text()
        assert parse_summary(text) is not None
        san = parse_sanitizer(text)
        assert san["banner"] and not san["headers"]
        assert (san["hazards"] if tool == "racecheck" else san["errors"]) == 0
        assert parse_gdn_cases(text)["cases"]["ok"] == expected
        if name != "mc-old":
            assert parse_engagement(text)["all_skipped"]
        supported = "\n".join(ANSI.sub("", line) for line in text.splitlines() if ": not supported" not in ANSI.sub("", line))
        assert not judge(tool, supported, text, expected, 0)
        assert judge(tool, supported, text, expected + 1, 0)
        assert "finding" in judge(tool, supported.replace(": OK", ": FAIL", 1), text, expected, 0)
        broken_san = text.replace("ERROR SUMMARY: 0 errors", "ERROR SUMMARY: 1 error").replace(
            "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)", "RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)")
        assert "finding" in judge(tool, supported, broken_san, expected, 0)
        if name != "mc-old":
            assert "path-engaged proof failed" in judge(tool, supported.replace("gdn_state_indexed:", "disabled:"), text, expected, 0)
    assert parse_xids((FIXTURES / "xid-lines.txt").read_text())
    assert not parse_xids("stop_xid_hash_s_G")
    assert parse_sanitizer("empty")["errors"] is None
    print("recorded GPU artifacts: parsers and negative controls PASS", flush=True)
    return 0


def check_frozen():
    subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=BIN, check=True)
    stamp = (ROOT / "SOURCE").read_text().splitlines()[0][:9].encode() + b"\0"
    assert any(stamp in p.read_bytes() for p in BIN.glob("libggml-base.so.*")), "ggml stamp mismatch"
    for exe in ["test-backend-ops"]:
        out = subprocess.check_output(["ldd", str(BIN / exe)], text=True)
        assert "not found" not in out
        for line in out.splitlines():
            if "libggml" in line:
                path = line.split("=>", 1)[1].strip().split()[0]
                assert Path(path).resolve().parent == BIN.resolve(), line


def journal(start):
    result = subprocess.run(["journalctl", "-k", "--no-pager", "-o", "short-unix", "--since", "@" + str(int(start))],
                            capture_output=True, text=True, timeout=10, check=True)
    return result.stdout


def execute(args):
    if os.environ.get("GPUQ_ACTIVE") != "1":
        raise ValueError("GPUQ_ACTIVE=1 required")
    check_frozen()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    budget = args.budget
    canary = args.stage == "canary"
    env = dict(os.environ, LD_LIBRARY_PATH=str(BIN), LD_PRELOAD=str(BIN / "backend-thread-cap.so"),
               GGML_CUDA_DEBUG_GDN_STATE="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    idx = "GATED_DELTA_NET_INDEXED,GATED_DELTA_NET_INDEXED_CACHE_FUSION"
    stages = [("mc-old", "memcheck", "GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION", 41, 240),
              ("mc-idx", "memcheck", idx, 16, 180), ("rc-idx", "racecheck", idx, 16, 180),
              ("sc-idx", "synccheck", idx, 16, 180)]
    if canary:
        stages = [("mc-old", "memcheck", "GATED_DELTA_NET", 1, 30),
                  ("mc-idx", "memcheck", idx, 3, 60), ("rc-idx", "racecheck", idx, 3, 60),
                  ("sc-idx", "synccheck", idx, 3, 60)]
    verdict = {"stages": [], "rc": 0}
    for name, tool, ops, count, cap in stages:
        left = budget - (time.time() - started) - 30
        if left < cap:
            verdict.update(rc=7, reason="budget")
            break
        target_path = args.out / (name + ".target.log")
        san_path = args.out / (name + ".sanitizer.log")
        cmd = [SANITIZER, "--tool", tool, "--log-file", str(san_path), "--error-exitcode", "99",
               "--print-limit", "50", "--report-api-errors", "no"]
        if tool == "racecheck":
            cmd += ["--racecheck-report", "all"]
        cmd += [str(BIN / "test-backend-ops"), "test", "-b", "CUDA0", "-j", "1", "-o", ops]
        if canary:
            pattern = "head_size=16,n_seq_tokens=1,n_seqs=1,.*kda=0,K=1" if name == "mc-old" else "head_size=32,n_seq_tokens=6,n_seqs=2,"
            cmd += ["-p", pattern]
        timeout = False
        with target_path.open("wb") as output:
            process = subprocess.Popen(cmd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                rc = 7
        target = target_path.read_text(errors="replace")
        san = san_path.read_text(errors="replace") if san_path.exists() else ""
        problems = judge(tool, target, san, count, rc)
        verdict["stages"].append({"name": name, "command": cmd, "rc": rc, "timeout": timeout, "problems": problems})
        # Save each stage before evaluating the run gates.
        (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        faults = journal(started)
        (args.out / "journal.txt").write_text(faults)
        if parse_xids(faults):
            verdict.update(rc=3, reason="kernel fault")
        elif timeout:
            verdict.update(rc=7, reason="budget")
        elif problems:
            verdict.update(rc=8 if "finding" in problems else 6, reason=problems)
        if verdict["rc"]:
            break
    verdict["wall_s_M"] = time.time() - started
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict), flush=True)
    return verdict["rc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["selftest", "preflight", "canary", "run"], required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget", type=int, default=1200)
    args = parser.parse_args()
    if args.stage == "selftest":
        return selftest()
    if args.stage == "preflight":
        check_frozen()
        return selftest()
    if args.out is None:
        parser.error("--out is required")
    os.nice(19)
    return execute(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print("integrity: " + str(error), flush=True)
        sys.exit(6)
