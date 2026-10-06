"""CPU evidence for GGML_META_AR_NODE: default-off identity vs the stack6 base, env-on gate on CPU, forced-node negative control.
Run through bin/cpugate with CUDA_VISIBLE_DEVICES="" after build_cpu_ar_node.sh. Writes codex/cpu-results/identity-check.log."""
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only"
RES = ROOT / "codex/cpu-results"
RUNS = RES / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("MODEL", "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf")
BASE = ROOT / "codex/base-cpu/ar-node-server-cpu"
SPIKE = ROOT / "build-cpu/bin/ar-node-server-cpu"
K = "3"
TOKENS = os.environ.get("TOKENS", "48")
SEEDS = ((0, 1729), (1, 1729), (1, 2718))
RESUME = os.environ.get("RESUME_META") == "1"  # continue a log after a failed check: skip the meta=0 cases already logged
LOG = (RES / "identity-check.log").open("a" if RESUME else "w")


def say(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def gate(max_s=4 * 3600):
    """A queued GPU item's timings are host-sensitive: start no run while one is RUNning (see AGENTS.md / bin/cpugate)."""
    t0 = time.monotonic()
    while re.search(r"\bRUN\b", subprocess.run(["gpuq", "status", "--brief"], capture_output=True, text=True).stdout):
        if time.monotonic() - t0 > max_s:
            say("FAIL: gpuq stayed busy past the wait cap")
            sys.exit(75)
        time.sleep(60)


def run(label, binary, meta, np_, temp, seed, extra=None, expect_fail=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_META", "LLAMA_"))}
    env.update(CPU_NP=str(np_), CPU_TEMP=str(temp), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
    env["LD_LIBRARY_PATH"] = str(Path(binary).parent)  # the base copy has no RUNPATH to its libs (its build dir is deleted)
    env.update(extra or {})
    out = RUNS / f"{label}.bin"
    gate()
    t0 = time.monotonic()
    with (RUNS / f"{label}.log").open("w") as log:
        r = subprocess.run(["nice", "-n", "19", str(binary), MODEL, str(out), str(meta), K, str(seed), TOKENS],
                           env=env, stdout=log, stderr=subprocess.STDOUT)
    text = (RUNS / f"{label}.log").read_text(errors="replace")
    if expect_fail:
        return r.returncode, text, None, time.monotonic() - t0
    if r.returncode != 0:
        say(f"FAIL {label}: exit {r.returncode}, see {RUNS / (label + '.log')}")
        sys.exit(1)
    data = out.read_bytes()
    return r.returncode, text, data, time.monotonic() - t0


PLAN = re.compile(r"meta_ar_node: uid=(\d+) sites=(\d+) split=(\d+)->(\d+) nodes=(\d+) rows=(-?\d+)")


def plans(text):
    return [tuple(int(x) for x in m.groups()) for m in PLAN.finditer(text)]


def main():
    say(f"model {MODEL}; draft n_max {K}; n_predict {TOKENS}; MTP draft = the target's own MTP head (as the existing CPU fixtures)")
    say(f"base binary {BASE} (stack6 ac7465885), spike binary {SPIKE}")
    kinds = collections.defaultdict(set)
    n_cases = 0
    for meta in ((1,) if RESUME else (0, 1)):
        for np_ in (1, 2):
            for temp, seed in SEEDS:
                tag = f"m{meta}-np{np_}-t{temp}-s{seed}"
                _, _, ref, w0 = run(f"{tag}-base", BASE, meta, np_, temp, seed)
                _, _, off, _ = run(f"{tag}-off", SPIKE, meta, np_, temp, seed)
                _, _, ref2, _ = run(f"{tag}-base2", BASE, meta, np_, temp, seed)
                rc, text_on, on, _ = run(f"{tag}-on", SPIKE, meta, np_, temp, seed, {"GGML_META_AR_NODE": "1"})
                if ref != ref2:
                    say(f"FAIL {tag}: base is not deterministic across two runs (control for (a))")
                    sys.exit(1)
                if off != ref:
                    say(f"FAIL {tag}: (a) default-off differs from base (sha {hashlib.sha256(off).hexdigest()[:12]} vs {hashlib.sha256(ref).hexdigest()[:12]})")
                    sys.exit(1)
                if on != ref:
                    say(f"FAIL {tag}: (b) env-on differs from base")
                    sys.exit(1)
                pl = plans(text_on)
                if meta:
                    if "meta_ar_node: unavailable (no backend node support)" not in text_on or not pl:
                        say(f"FAIL {tag}: (b) env-on log lacks the unavailable line or the would-be plan lines")
                        sys.exit(1)
                    # once per Meta context: the target and, with MTP, the draft context
                    if not 1 <= text_on.count("meta_ar_node: unavailable") <= 2:
                        say(f"FAIL {tag}: (b) unavailable line must be logged once per Meta context")
                        sys.exit(1)
                    for uid, sites, m, k, nodes, rows in pl:
                        kinds[rows].add((sites, m, k, nodes))
                elif pl or "meta_ar_node" in text_on:
                    say(f"FAIL {tag}: meta_ar_node logged without a meta backend")
                    sys.exit(1)
                if "meta_ar_node" in run_text(f"{tag}-off"):
                    say(f"FAIL {tag}: meta_ar_node logged with the env unset")
                    sys.exit(1)
                n_cases += 1
                say(f"PASS {tag}: base == base2 == off == on, {len(ref)} bytes, sha256 {hashlib.sha256(ref).hexdigest()[:16]}, "
                    f"wall base {w0:.0f}s; plan lines {len(pl)}")
    # negative control for the identity comparison itself
    altered = bytearray(ref)
    altered[8] ^= 1
    assert bytes(altered) != ref
    say("PASS negative control (a)/(b): a one-bit perturbed token stream is rejected by the comparison")

    say("would-be node plans on CPU (Meta(CPU,CPU)), per graph kind (rows = tokens per graph): (sites, split M, nodes K, cgraph nodes)")
    for rows in sorted(kinds):
        say(f"  rows={rows}: {sorted(kinds[rows])}")

    # (d) forced insertion: the stub must abort with the clear message
    rc, text, _, _ = run("forced-node", SPIKE, 1, 1, 0, 1729, {"GGML_META_AR_NODE": "1", "GGML_META_AR_NODE_FORCE_TEST": "1"}, expect_fail=True)
    if rc == 0 or "meta_ar_node: AllReduce node allreduce node" not in text or "executed by a backend without node support" not in text:
        say(f"FAIL (d) forced node insertion did not abort with the stub message (exit {rc})")
        sys.exit(1)
    say(f"PASS (d) forced node insertion on CPU aborts in the stub: exit {rc}; " +
        [l for l in text.splitlines() if "executed by a backend without node support" in l][0].strip())
    rc, text, _, _ = run("forced-node-env-off", SPIKE, 1, 1, 0, 1729, {"GGML_META_AR_NODE_FORCE_TEST": "1"}, expect_fail=True)
    if rc != 0:
        say(f"FAIL (d) the test hook acts without GGML_META_AR_NODE=1 (exit {rc})")
        sys.exit(1)
    say("PASS (d) the hook is inert without GGML_META_AR_NODE=1")
    say(f"SUMMARY: {n_cases} cases (meta 0/1 x np 1/2 x 3 sampling configs), each 4 runs, all byte-identical")


def run_text(label):
    return (RUNS / f"{label}.log").read_text(errors="replace")


if __name__ == "__main__":
    main()
