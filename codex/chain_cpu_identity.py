"""CPU identity of the MTP draft chain (--spec-mtp-chain) against the step loop, through the chain-cpu fixture (never llama-server).

T1: for meta in (plain CPU, Meta(CPU,CPU)) and k in (3, 5): the chain-off and chain-on boots must agree on every case's generated
token ids, per-round drafted ids / candidate ids / q (LLAMA_MTP_CHAIN_DUMP) and draft/accept counters.
T2: negative controls on top of (plain, k=5, chain on): reversed uniforms, shifted compact table; off-vs-off repeat as the noise floor.
Every boot goes through cpugate after gpuq has no running item; CUDA is hidden. Usage: chain_cpu_identity.py [--only T1|T2|SMOKE]"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "export CUDA_VISIBLE_DEVICES="
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "codex/cpu-results"
BIN = ROOT / "build-cpu/bin"
MODEL = os.environ.get("S7_MODEL", "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf")
VOCAB = os.environ.get("S7_VOCAB", "/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json")
CPUGATE = os.environ.get("S7_GATE", "/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate")

SHORT = "The sequence of positive integers begins with 1, 2, 3, 4, 5. Continue the sequence:"
MID = " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(25)) + "\nSummarize the list above in one sentence."
LONG = " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(100)) + "\nSummarize the list above in one sentence."
PROMPTS = {"short": SHORT, "mid": MID, "long": LONG}
RESUME = False
CELLS = [(0.0, 1729), (1.0, 1729), (1.0, 2718)]


def cases(n_predict, prompts=PROMPTS, cells=CELLS):
    return [{"name": f"{pn}-t{t:g}-s{s}", "prompt": p, "temp": t, "seed": s, "n_predict": n_predict}
            for pn, p in prompts.items() for t, s in cells]


def wait_gpuq_idle():
    if os.environ.get("GPUQ_BUILD_WINDOW") == "1":
        return
    while True:
        out = subprocess.run(["gpuq", "status", "--brief"], capture_output=True, text=True).stdout
        if not re.search(r"^gpuq: RUN", out):
            return
        print("gpuq busy:", out.strip(), flush=True)
        time.sleep(60)


def boot(tag, k, meta, chain, case_list, extra_env=None):
    d = RES / tag
    if RESUME and (d / "out.json").exists():
        return load(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cases.json").write_text(json.dumps(case_list))
    threads = "4" if meta else "8"
    cmd = [str(BIN / "chain-cpu"), "-m", MODEL, "-c", "4096", "-np", "1", "-ngl", "0", "-fa", "on", "-b", "256", "-ub", "256",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", str(k), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
           "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling",
           "--spec-draft-vocab", VOCAB, "--kv-pad", "4096", "-t", threads, "-tb", threads]
    if chain:
        cmd.append("--spec-mtp-chain")
    env = dict(os.environ, LD_LIBRARY_PATH=str(BIN), CHAIN_CASES=str(d / "cases.json"), CHAIN_OUT=str(d / "out.json"),
               CHAIN_META="1" if meta else "0", LLAMA_MTP_CHAIN_DUMP=str(d / "dump.txt"), LLAMA_ROUND_TRACE=str(d / "trace.jsonl"),
               LLAMA_ARG_LOG_VERBOSITY="3", OMP_NUM_THREADS=threads)
    for f in ("out.json", "dump.txt", "trace.jsonl"):
        (d / f).unlink(missing_ok=True)
    env.update(extra_env or {})
    wait_gpuq_idle()
    t0 = time.time()
    with (d / "log.txt").open("w") as log:
        rc = subprocess.run([CPUGATE] + cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"boot {tag} rc={rc} elapsed_s_M={time.time() - t0:.0f}", flush=True)
    assert rc == 0, f"boot {tag} failed, see {d / 'log.txt'}"
    return load(d)


def load(d):
    out = json.loads((d / "out.json").read_text())
    dump = [l.split() for l in (d / "dump.txt").read_text().splitlines()]
    counters = {}
    for line in (d / "trace.jsonl").read_text().splitlines():
        for ph in json.loads(line)["phases"]:
            for c in ph["counters"]:
                if c["kind"] in ("graph_hit", "graph_miss", "mtp_chain", "mtp_fold"):
                    key = (ph["phase"] if ph["phase"][0] != "D" or ph["phase"] == "DC" else "D", c["kind"])
                    counters[key] = counters.get(key, 0) + c["count"]
    log = (d / "log.txt").read_text()
    return {"out": out, "dump": dump, "counters": counters, "chain_line": len(re.findall(r"MTP chain: k=\d+", log)),
            "log": str(d / "log.txt")}


def compare(a, b):
    """a, b: boot results. Returns per-case token identity and dump comparison."""
    res = {"tokens": {}, "counters_equal": {}, "n_tokens": 0}
    for ca, cb in zip(a["out"], b["out"]):
        assert ca["name"] == cb["name"]
        res["tokens"][ca["name"]] = ca["tokens"] == cb["tokens"]
        res["counters_equal"][ca["name"]] = (ca["drafts"], ca["draft_tokens"], ca["accepted_tokens"]) == (cb["drafts"], cb["draft_tokens"], cb["accepted_tokens"])
        res["n_tokens"] += len(ca["tokens"])
    res["dump_rows"] = (len(a["dump"]), len(b["dump"]))
    ids_equal = qs_equal = True
    max_dq = 0.0
    n_ids_diff = 0
    for ra, rb in zip(a["dump"], b["dump"]):
        if ra[:13] != rb[:13]:
            ids_equal = False
            n_ids_diff += 1
        if ra[13:] != rb[13:]:
            qs_equal = False
            max_dq = max(max_dq, max(abs(float(x) - float(y)) for x, y in zip(ra[13:], rb[13:])))
    res.update(dump_ids_equal=ids_equal and len(a["dump"]) == len(b["dump"]), dump_q_bit_equal=qs_equal, dump_max_abs_dq=max_dq,
               dump_rows_ids_differ=n_ids_diff)
    return res


def summarize(out):
    return {c["name"]: (len(c["tokens"]), c["prompt_n"], c["drafts"], c["draft_tokens"], c["accepted_tokens"]) for c in out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="all")
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--resume", action="store_true", help="reuse boots whose out.json exists")
    args = ap.parse_args()
    global RESUME, RES
    RES = args.out.resolve()
    RESUME = args.resume
    RES.mkdir(parents=True, exist_ok=True)
    rp = RES / "report.json"
    report = json.loads(rp.read_text()) if args.resume and rp.exists() else {}

    if args.only == "SMOKE":
        cl = cases(24, {"short": SHORT}, [(1.0, 1729)])
        off = boot("smoke-off", 3, False, False, cl)
        on = boot("smoke-on", 3, False, True, cl)
        print(json.dumps(compare(off, on), indent=1), on["counters"], on["chain_line"])
        return

    cl = cases(args.n_predict)
    if args.only in ("all", "T1"):
        for meta in (False, True):
            for k in (3, 5):
                tag = f"{'meta' if meta else 'plain'}-k{k}"
                off = boot(f"{tag}-off", k, meta, False, cl)
                on = boot(f"{tag}-on", k, meta, True, cl)
                cmp = compare(off, on)
                report[tag] = {"cmp": cmp, "off_counters": {f"{a}/{b}": v for (a, b), v in off["counters"].items()},
                               "on_counters": {f"{a}/{b}": v for (a, b), v in on["counters"].items()},
                               "off_chain_lines": off["chain_line"], "on_chain_lines": on["chain_line"],
                               "off": summarize(off["out"]), "on": summarize(on["out"])}
                (RES / "report.json").write_text(json.dumps(report, indent=1))
                print(tag, json.dumps(cmp), flush=True)

    if args.only in ("all", "T2"):
        d_off = RES / "plain-k5-off"
        base_off = load(d_off) if (d_off / "out.json").exists() else boot("plain-k5-off", 5, False, False, cl)
        rep = boot("plain-k5-off-repeat", 5, False, False, cl)
        report["noise_floor_off_vs_off"] = compare(base_off, rep)
        rev = boot("plain-k5-on-reverse-u", 5, False, True, cl, {"LLAMA_MTP_CHAIN_DEBUG_REVERSE_U": "1"})
        report["control_reverse_u"] = {"cmp": compare(base_off, rev), "on": summarize(rev["out"])}
        shift = boot("plain-k5-on-shift-embd", 5, False, True, cl, {"LLAMA_MTP_CHAIN_DEBUG_SHIFT_EMBD": "1"})
        report["control_shift_embd"] = {"cmp": compare(base_off, shift), "on": summarize(shift["out"])}
        (RES / "report.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: report[k] for k in report if k.startswith(("noise", "control"))}, indent=1))


if __name__ == "__main__":
    main()
