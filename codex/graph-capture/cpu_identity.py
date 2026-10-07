"""CPU identity for GGML_META_AR_NODE on stack7: base (stack7) vs HEAD with the env unset and set, Meta(CPU,CPU) and plain CPU.

usage: cpu_identity.py BASE_FIXTURE HEAD_FIXTURE MODEL OUT_DIR
Gates (exit 2 on any failure):
  (a) HEAD env unset == base, byte for byte (tokens of every slot + final target logits); base == base2 is the control.
  (b) HEAD GGML_META_AR_NODE=1 == base. On CPU node mode is unavailable: the log must say so and print would-be plans.
  (c) negative control: a one-bit perturbed stream is rejected by the comparison; the base fixture never logs meta_ar_node.
  (d) GGML_META_AR_NODE_FORCE_TEST=1 inserts nodes on CPU and must abort in the stub; without GGML_META_AR_NODE it is inert.
"""
import collections
import hashlib
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

BASE, HEAD, MODEL, OUT = sys.argv[1:5]
OUT = Path(OUT)
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
LOG = (OUT / "identity.log").open("w")
K, TOKENS = "3", os.environ.get("TOKENS", "48")
CHAIN_FOLD = {"LLAMA_MTP_CHAIN": "1", "LLAMA_MTP_FOLD_C": "1"}
# (meta, np, temp, seed, extra env): np1 chain+fold is the np1 serving shape, np2 runs chain/fold off as the duo profile
CASES = [
    (1, 1, 0, 1729, {}), (1, 1, 1, 1729, {}), (1, 1, 1, 2718, {}),
    (1, 1, 0, 1729, CHAIN_FOLD), (1, 1, 1, 1729, CHAIN_FOLD),
    (1, 2, 0, 1729, {}), (1, 2, 1, 1729, {}),
    (0, 1, 1, 1729, {}),
]
PLAN = re.compile(r"meta_ar_node: uid=(\d+) sites=(\d+) split=(\d+)->(\d+) nodes=(\d+) rows=(-?\d+)")
MODEL_PROCS = {"ar-node-server-cpu", "bw-cand-driver", "bw-np2-driver", "chain-cpu", "fold-cpu", "outproj-cpu"}


def say(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def fail(msg):
    say("FAIL " + msg)
    sys.exit(2)


def wait_for_room():
    """Shared Mac: one model process at a time and >= 4 GiB available."""
    if platform.system() != "Darwin":
        return
    while True:
        memory = subprocess.check_output(["memory_pressure", "-Q"], text=True)
        free = int(re.search(r"free percentage: (\d+)%", memory)[1]) / 100
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        comms = subprocess.check_output(["ps", "-axo", "comm"], text=True).splitlines()
        busy = any(c.rsplit("/", 1)[-1] in MODEL_PROCS for c in comms)
        if not busy and total * free >= 4 * 1024**3:
            return
        say(f"WAIT busy={busy} free={total * free / 1024**3:.1f} GiB")
        time.sleep(20)


def run(label, binary, meta, np_, temp, seed, extra, expect_fail=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_META", "LLAMA_", "GGML_CUDA"))}
    env.update(CUDA_VISIBLE_DEVICES="", CPU_NP=str(np_), CPU_TEMP=str(temp), CPU_THREADS="4", OMP_NUM_THREADS="4")
    env.update(extra)
    out = RUNS / f"{label}.bin"
    log = RUNS / f"{label}.log"
    wait_for_room()
    t0 = time.monotonic()
    with log.open("w") as stream:
        r = subprocess.run(["nice", "-n", "19", binary, MODEL, str(out), str(meta), K, str(seed), TOKENS],
                           env=env, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(errors="replace")
    if expect_fail:
        return r.returncode, text, None
    if r.returncode != 0:
        fail(f"{label}: exit {r.returncode}, see {log}")
    say(f"ran {label} in {time.monotonic() - t0:.0f}s")
    return r.returncode, text, out.read_bytes()


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    say(f"model {MODEL}; draft n_max {K}; n_predict {TOKENS}; base {BASE}; head {HEAD}")
    kinds = collections.defaultdict(set)
    last = None
    for meta, np_, temp, seed, extra in CASES:
        tag = f"m{meta}-np{np_}-t{temp}-s{seed}" + ("-chainfold" if extra else "")
        _, t_base, ref = run(f"{tag}-base", BASE, meta, np_, temp, seed, extra)
        _, t_off, off = run(f"{tag}-off", HEAD, meta, np_, temp, seed, extra)
        _, _, ref2 = run(f"{tag}-base2", BASE, meta, np_, temp, seed, extra)
        _, t_on, on = run(f"{tag}-on", HEAD, meta, np_, temp, seed, {**extra, "GGML_META_AR_NODE": "1"})
        if ref != ref2:
            fail(f"{tag}: base not deterministic across two runs (control for (a))")
        if off != ref:
            fail(f"{tag}: (a) env unset differs from base ({sha(off)} vs {sha(ref)})")
        if on != ref:
            fail(f"{tag}: (b) GGML_META_AR_NODE=1 differs from base ({sha(on)} vs {sha(ref)})")
        if "meta_ar_node" in t_base or "meta_ar_node" in t_off:
            fail(f"{tag}: meta_ar_node logged by the base or with the env unset")
        plans = [tuple(map(int, m.groups())) for m in PLAN.finditer(t_on)]
        if meta:
            n_unavail = t_on.count("meta_ar_node: unavailable (no backend node support)")
            if not plans or not 1 <= n_unavail <= 2:  # once per Meta context: target, and the draft with MTP
                fail(f"{tag}: (b) env-on log lacks the would-be plans or logs 'unavailable' {n_unavail} times")
            for _, sites, m, k, nodes, rows in plans:
                kinds[rows].add((sites, m, k, nodes))
        elif "meta_ar_node" in t_on:
            fail(f"{tag}: meta_ar_node logged without a Meta backend")
        engaged = [s for s in ("MTP chain: k=", "MTP fold: on", "mtp_fold_decode: engaged") if s in t_on]
        say(f"PASS {tag}: base == base2 == off == on, {len(ref)} bytes, sha {sha(ref)}; plan lines {len(plans)}; "
            f"chain/fold log lines {engaged or 'none'}")
        last = ref
    altered = bytearray(last)
    altered[8] ^= 1
    if bytes(altered) == last:
        fail("negative control: perturbation did not change the stream")
    say("PASS (c) a one-bit perturbed token stream is rejected by the comparison; base never logs meta_ar_node")
    say("would-be node plans on Meta(CPU,CPU) per graph kind (rows): (sites, split M, subgraphs K, cgraph nodes)")
    for rows in sorted(kinds):
        say(f"  rows={rows}: {sorted(kinds[rows])}")

    force = {"GGML_META_AR_NODE": "1", "GGML_META_AR_NODE_FORCE_TEST": "1"}
    rc, text, _ = run("forced-node", HEAD, 1, 1, 0, 1729, force, expect_fail=True)
    if rc == 0 or "executed by a backend without node support" not in text:
        fail(f"(d) forced node insertion did not abort in the stub (exit {rc})")
    say(f"PASS (d) forced nodes abort in the stub, exit {rc}: " +
        next(l for l in text.splitlines() if "executed by a backend without node support" in l).strip())
    rc, text, _ = run("forced-node-env-off", HEAD, 1, 1, 0, 1729, {"GGML_META_AR_NODE_FORCE_TEST": "1"}, expect_fail=True)
    if rc != 0 or "meta_ar_node" in text:
        fail(f"(d) the test hook acts without GGML_META_AR_NODE=1 (exit {rc})")
    say("PASS (d) the hook is inert without GGML_META_AR_NODE=1")
    say(f"SUMMARY PASS: {len(CASES)} cases x 4 runs byte-identical; controls (c), (d) pass")


if __name__ == "__main__":
    main()
