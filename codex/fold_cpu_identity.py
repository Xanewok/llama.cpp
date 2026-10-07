"""CPU identity of the MTP fold (LLAMA_MTP_FOLD_C=1) against the stack6 catch-up decode, through the fold-cpu fixture (never
llama-server). Usage: fold_cpu_identity.py OUT_DIR [--only T1|T2|SMOKE] [--n-predict N]

T1: plain CPU and Meta(CPU,CPU) x k 3/5: fold off vs on must agree on every case's tokens, draft counters and per-step
    drafted ids / candidate ids / probabilities (LLAMA_MTP_DRAFT_DUMP, %.9g strings, compared as strings). Engagement:
    on has "MTP fold: on" and mtp_fold > 0 with fewer C phases; off has neither.
T2: off-vs-off repeat must be identical (noise floor); the fold mutant (row i paired with nextn row i) must differ from off.
Exit: 0 pass | 2 identity failure | 4 mutant escaped | 5 off not repeatable | 7 engagement | 8 boot error.
Boots go through cpugate (waits out measure items); CUDA is hidden."""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "export CUDA_VISIBLE_DEVICES="
ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build-cpu/bin"
MODEL = "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf"
VOCAB = "/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"
CPUGATE = "/home/xanewok/repos/qwen38-eval-worktrees/bin/cpugate"
FOLD_ENVS = ("LLAMA_MTP_FOLD_C", "LLAMA_MTP_FOLD_DEBUG_MUTANT")

# short (< 64 tokens) also exercises the folded prompt decode from the second case on; mid/long prefill in 256-token ubatches
SHORT = "The sequence of positive integers begins with 1, 2, 3, 4, 5. Continue the sequence:"
MID = " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(25)) + "\nSummarize the list above in one sentence."
LONG = " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}." for i in range(100)) + "\nSummarize the list above in one sentence."
PROMPTS = {"short": SHORT, "mid": MID, "long": LONG}
CELLS = [(0.0, 1729), (1.0, 1729), (1.0, 2718)]


class Verdict(Exception):
    def __init__(self, rc, msg):
        super().__init__(msg)
        self.rc = rc


def cases(n_predict, prompts=PROMPTS, cells=CELLS):
    return [{"name": f"{pn}-t{t:g}-s{s}", "prompt": p, "temp": t, "seed": s, "n_predict": n_predict}
            for pn, p in prompts.items() for t, s in cells]


def boot(res, tag, k, meta, fold, case_list, extra_env=None):
    d = res / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "cases.json").write_text(json.dumps(case_list))
    threads = "4" if meta else "8"
    cmd = [str(BIN / "fold-cpu"), "-m", MODEL, "-c", "4096", "-np", "1", "-ngl", "0", "-fa", "on", "-b", "256", "-ub", "256",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", str(k), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
           "--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling",
           "--spec-draft-vocab", VOCAB, "--kv-pad", "4096", "-t", threads, "-tb", threads]
    env = {k_: v for k_, v in os.environ.items() if k_ not in FOLD_ENVS}
    env.update(LD_LIBRARY_PATH=str(BIN), FOLD_CASES=str(d / "cases.json"), FOLD_OUT=str(d / "out.json"),
               FOLD_META="1" if meta else "0", LLAMA_MTP_DRAFT_DUMP=str(d / "dump.txt"), LLAMA_ROUND_TRACE=str(d / "trace.jsonl"),
               LLAMA_ARG_LOG_VERBOSITY="4", GGML_SCHED_INPUT_STAGING="1", OMP_NUM_THREADS=threads)
    if fold:
        env["LLAMA_MTP_FOLD_C"] = "1"
    env.update(extra_env or {})
    for f in ("out.json", "dump.txt", "trace.jsonl"):
        (d / f).unlink(missing_ok=True)
    t0 = time.time()
    with (d / "log.txt").open("w") as log:
        rc = subprocess.run([CPUGATE] + cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"boot {tag} rc={rc} elapsed_s_M={time.time() - t0:.0f}", flush=True)
    if rc != 0:
        raise Verdict(8, f"boot {tag} failed, see {d / 'log.txt'}")
    return load(d)


def load(d):
    out = json.loads((d / "out.json").read_text())
    dump = [l.split() for l in (d / "dump.txt").read_text().splitlines()]
    counters, phases = {}, {}
    for line in (d / "trace.jsonl").read_text().splitlines():
        for ph in json.loads(line)["phases"]:
            name = ph["phase"] if ph["phase"][0] != "D" else "D"
            phases[name] = phases.get(name, 0) + 1
            if ph["phase"] == "D1":
                phases["D1"] = phases.get("D1", 0) + 1
            for c in ph["counters"]:
                if c["kind"] in ("graph_hit", "graph_miss", "mtp_fold"):
                    counters[f"{name}/{c['kind']}"] = counters.get(f"{name}/{c['kind']}", 0) + c["count"]
    log = (d / "log.txt").read_text()
    return {"out": out, "dump": dump, "counters": counters, "phases": phases,
            "fold_on": len(re.findall(r"MTP fold: on\b", log)), "fold_off": len(re.findall(r"MTP fold: off\b", log)),
            "ignored": len(re.findall(r"LLAMA_MTP_FOLD_C ignored", log)), "log": str(d / "log.txt")}


def mtp_fold(r):
    return sum(v for key, v in r["counters"].items() if key.endswith("/mtp_fold"))


def compare(a, b):
    res = {"tokens": {}, "counters_equal": {}, "n_tokens": 0}
    for ca, cb in zip(a["out"], b["out"]):
        assert ca["name"] == cb["name"]
        res["tokens"][ca["name"]] = ca["tokens"] == cb["tokens"]
        res["counters_equal"][ca["name"]] = (ca["drafts"], ca["draft_tokens"], ca["accepted_tokens"]) == (cb["drafts"], cb["draft_tokens"], cb["accepted_tokens"])
        res["n_tokens"] += len(ca["tokens"])
    res["n_cases"] = (len(a["out"]), len(b["out"]))
    res["dump_rows"] = (len(a["dump"]), len(b["dump"]))
    n_ids_diff = n_q_diff = 0
    for ra, rb in zip(a["dump"], b["dump"]):
        n_ids_diff += ra[:13] != rb[:13]
        n_q_diff += ra[13:] != rb[13:]
    res.update(dump_rows_ids_differ=n_ids_diff, dump_rows_q_differ=n_q_diff)
    res["identical"] = (res["n_cases"][0] == res["n_cases"][1] and all(res["tokens"].values()) and all(res["counters_equal"].values())
                        and res["dump_rows"][0] == res["dump_rows"][1] and res["dump_rows"][0] > 0 and n_ids_diff == 0 and n_q_diff == 0)
    return res


def summarize(r):
    return {"cases": {c["name"]: (len(c["tokens"]), c["prompt_n"], c["drafts"], c["draft_tokens"], c["accepted_tokens"]) for c in r["out"]},
            "counters": r["counters"], "phases": r["phases"], "fold_on_lines": r["fold_on"], "fold_off_lines": r["fold_off"],
            "ignored_lines": r["ignored"]}


def engaged(tag, off, on):
    """on: canary line, folds, fewer catch-up decodes; off: no fold at all."""
    errs = []
    if on["fold_on"] < 1:
        errs.append(f"{tag}: no 'MTP fold: on' line")
    if mtp_fold(on) < 1:
        errs.append(f"{tag}: mtp_fold counter is 0 with the fold on")
    # Every draft round but each request's first follows a verify; a fold refused on verify rounds falls back to C
    # silently and would still pass the identity checks.
    rounds = on["phases"].get("D1", 0)
    if mtp_fold(on) < 0.9 * rounds:
        errs.append(f"{tag}: mtp_fold {mtp_fold(on)} < 0.9 x draft rounds {rounds}")
    if on["phases"].get("C", 0) >= off["phases"].get("C", 0):
        errs.append(f"{tag}: C phases on={on['phases'].get('C', 0)} not below off={off['phases'].get('C', 0)}")
    if off["fold_on"] or off["fold_off"] or mtp_fold(off):
        errs.append(f"{tag}: the off boot folded or logged a fold line")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", default="all")
    ap.add_argument("--n-predict", type=int, default=96)
    args = ap.parse_args()
    res = args.out.resolve()
    res.mkdir(parents=True, exist_ok=True)
    report, fails = {}, []

    def save():
        (res / "report.json").write_text(json.dumps(report, indent=1))

    try:
        if args.only == "SMOKE":
            cl = cases(24, {"short": SHORT}, [(1.0, 1729)]) * 2
            for i, c in enumerate(cl):
                c["name"] += f"-{i}"
            off = boot(res, "smoke-off", 3, False, False, cl)
            on = boot(res, "smoke-on", 3, False, True, cl)
            report["smoke"] = {"cmp": compare(off, on), "off": summarize(off), "on": summarize(on), "engagement": engaged("smoke", off, on)}
            save()
            print(json.dumps(report["smoke"], indent=1))
            return 0 if report["smoke"]["cmp"]["identical"] and not report["smoke"]["engagement"] else 2

        cl = cases(args.n_predict)
        base_off = None
        if args.only in ("all", "T1"):
            for meta in (False, True):
                for k in (3, 5):
                    tag = f"{'meta' if meta else 'plain'}-k{k}"
                    off = boot(res, f"{tag}-off", k, meta, False, cl)
                    on = boot(res, f"{tag}-on", k, meta, True, cl)
                    if tag == "plain-k5":
                        base_off = off
                    cmp = compare(off, on)
                    eng = engaged(tag, off, on)
                    report[tag] = {"cmp": cmp, "engagement": eng, "off": summarize(off), "on": summarize(on)}
                    save()
                    print(tag, json.dumps({"identical": cmp["identical"], "engagement": eng, "on_mtp_fold": mtp_fold(on),
                                           "C_off": off["phases"].get("C", 0), "C_on": on["phases"].get("C", 0)}), flush=True)
                    if not cmp["identical"]:
                        fails.append((2, f"{tag}: fold on differs from off"))
                    fails += [(7, e) for e in eng]

        if args.only in ("all", "T2"):
            if base_off is None:
                base_off = boot(res, "plain-k5-off", 5, False, False, cl)
            rep = boot(res, "plain-k5-off-repeat", 5, False, False, cl)
            report["noise_floor_off_vs_off"] = compare(base_off, rep)
            if not report["noise_floor_off_vs_off"]["identical"]:
                fails.append((5, "off vs off repeat differs: the reference is not repeatable, T1 proves nothing"))
            mut = boot(res, "plain-k5-on-mutant", 5, False, True, cl, {"LLAMA_MTP_FOLD_DEBUG_MUTANT": "1"})
            report["control_mutant"] = {"cmp": compare(base_off, mut), "on": summarize(mut)}
            if mtp_fold(mut) < 1:
                fails.append((4, "mutant did not fold, so it cannot test anything"))
            elif report["control_mutant"]["cmp"]["identical"]:
                fails.append((4, "mutant (row i paired with nextn row i) matched off: the identity test cannot see a wrong pairing"))
            save()
            print(json.dumps({k: report[k] for k in report if k.startswith(("noise", "control"))}, indent=1))
    except Verdict as v:
        fails.append((v.rc, str(v)))

    report["fails"] = fails
    save()
    for rc, msg in fails:
        print(f"FAIL rc={rc}: {msg}", flush=True)
    return min((rc for rc, _ in fails), key=lambda rc: [8, 5, 4, 2, 7].index(rc)) if fails else 0


if __name__ == "__main__":
    sys.exit(main())
