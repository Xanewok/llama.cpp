"""Queue item 30 (llama-agg) client and summary: aggregate decode of one booted llama-server, 1 or 2 slots.

Prompts are fixed across arms and boots: frozen corpus.txt (never the live repo), cut by tokens, sent as one user
message and rendered by the server's own chat template (reasoning on, default effort); their sha256 is recorded.
Stream s reads corpus tokens from s * OFFSET, so the two slots share no prefix. Depths are nested per stream, so a
slot prefills only the increment over its previous depth (from the checkpoint the server leaves n_ubatch + 4 tokens
before each prompt end). Per depth: each slot prefills alone (n_predict 1), then REPS rounds in which every slot
streams N_PREDICT tokens concurrently from a cache hit (the server replays ~4 tokens).
Aggregate = tokens of all streams inside the overlap window [max first token, min last token] / window length.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTION = ("Summarize what the code and notes above do, then list the three most likely bugs, "
            "each with the file and the reason.")
OFFSET = 100000  # tokens between stream starts; > max depth, so streams never share a prefix
MISS_TOKENS = 1000  # a timed request that re-evaluates more than this missed its checkpoint


def call(url, path, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def mem_available_mib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def sampling(seed):
    return {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05, "seed": seed}


def build_prompts(url, n_streams, depths):
    corpus = (HERE / "corpus.txt").read_text()
    tokens = call(url, "/tokenize", {"content": corpus})["tokens"]
    need = (n_streams - 1) * OFFSET + max(depths)
    while len(tokens) < need:
        tokens += tokens
    empty = call(url, "/apply-template", {"messages": [{"role": "user", "content": "\n\n" + QUESTION}]})["prompt"]
    overhead = len(call(url, "/tokenize", {"content": empty})["tokens"])
    prompts = {}
    for s in range(n_streams):
        for d in depths:
            chunk = call(url, "/detokenize", {"tokens": tokens[s * OFFSET: s * OFFSET + d - overhead]})["content"]
            p = call(url, "/apply-template", {"messages": [{"role": "user", "content": chunk + "\n\n" + QUESTION}]})
            prompts[(s, d)] = p["prompt"]
    return prompts, {"corpus_tokens": len(tokens), "template_overhead_tokens": overhead}


def prefill(url, prompt, slot, seed):
    t0 = time.monotonic()
    r = call(url, "/completion", {"prompt": prompt, "n_predict": 1, "cache_prompt": True, "id_slot": slot,
                                  **sampling(seed)})
    t = r.get("timings", {})
    return {"slot": slot, "wall_s": time.monotonic() - t0, "tokens_evaluated": r.get("tokens_evaluated"),
            "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"), "cache_n": t.get("cache_n"),
            "prompt_per_second": t.get("prompt_per_second")}


def stream(url, prompt, slot, seed, n_predict, out):
    body = {"prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "cache_prompt": True, "id_slot": slot,
            "stream": True, **sampling(seed)}
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    stamps, final, text = [], {}, []
    t_submit = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                c = json.loads(line[6:])
                if "error" in c:
                    raise RuntimeError(str(c["error"]))
                now = time.monotonic()
                if c.get("content"):
                    text.append(c["content"])
                if c.get("tokens_predicted"):
                    stamps.append((now, c["tokens_predicted"]))
                if c.get("stop"):
                    final = c
                    break
    except Exception as e:  # recorded; the round is dropped from the sample
        out["error"] = repr(e)
    content = "".join(text)
    out.update(slot=slot, seed=seed, t_submit=t_submit, stamps=stamps, timings=final.get("timings", {}),
               content_head=content[:160], think_closed="</think>" in content,
               eos_text="<|im_end|>" in content)


def n_at(stamps, t):
    n = 0
    for s, k in stamps:
        if s > t:
            break
        n = k
    return n


def round_metrics(outs):
    per = []
    for o in outs:
        st, t = o["stamps"], o["timings"]
        if o.get("error") or len(st) < 2:
            return {"error": o.get("error") or "no tokens", "streams": per}
        dur = st[-1][0] - st[0][0]
        per.append({"slot": o["slot"], "seed": o["seed"], "n": st[-1][1], "ttft_s": st[0][0] - o["t_submit"],
                    "decode_s": dur, "tg": (st[-1][1] - st[0][1]) / dur if dur > 0 else None,
                    "server_tg": t.get("predicted_per_second"), "replay_prompt_n": t.get("prompt_n"),
                    "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
                    "think_closed": o["think_closed"], "eos_text": o["eos_text"], "content_head": o["content_head"]})
    w0 = max(o["stamps"][0][0] for o in outs)
    w1 = min(o["stamps"][-1][0] for o in outs)
    inwin = sum(n_at(o["stamps"], w1) - n_at(o["stamps"], w0) for o in outs)
    u0 = min(o["stamps"][0][0] for o in outs)
    u1 = max(o["stamps"][-1][0] for o in outs)
    union = sum(o["stamps"][-1][1] - o["stamps"][0][1] for o in outs)
    return {"streams": per, "overlap_s": w1 - w0, "overlap_frac": (w1 - w0) / max(p["decode_s"] for p in per),
            "agg_tg": inwin / (w1 - w0) if w1 > w0 else None, "agg_union_tg": union / (u1 - u0) if u1 > u0 else None,
            "draft_n": sum(p["draft_n"] for p in per), "draft_accepted": sum(p["draft_accepted"] for p in per),
            "checkpoint_miss": any((p["replay_prompt_n"] or 0) > MISS_TOKENS for p in per)}


def boot(a):
    url = f"http://127.0.0.1:{a.port}"
    depths = [int(x) for x in a.depths.split(",")]
    res = {"arm": a.arm, "np": a.np, "k": a.k, "ctx": a.ctx, "boot": a.boot, "round": a.round,
           "depths": depths, "reps": a.reps, "n_predict": a.n_predict, "complete": False, "results": []}
    if a.meta and Path(a.meta).exists():
        res["launch"] = json.loads(Path(a.meta).read_text())

    def save():
        tmp = Path(str(a.out) + ".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, a.out)

    prompts, info = build_prompts(url, a.np, depths)
    res["prompt_info"] = {**info, "question": QUESTION, "offset": OFFSET,
                          "sha256": {f"s{s}-d{d}": hashlib.sha256(p.encode()).hexdigest()[:16]
                                     for (s, d), p in prompts.items()}}
    save()
    for di, d in enumerate(depths):
        if time.time() > a.deadline:
            res["stopped"] = f"deadline before depth {d}"
            break
        if mem_available_mib() < 2048:
            res["stopped"] = "MemAvailable < 2 GiB"
            save()
            sys.exit(5)
        row = {"depth": d, "prefill": [], "rounds": []}
        res["results"].append(row)
        for s in range(a.np):
            row["prefill"].append(prefill(url, prompts[(s, d)], s, 1729 + s))
            save()
        for rep in range(a.reps):
            if time.time() > a.deadline:
                res["stopped"] = f"deadline in depth {d}"
                break
            outs = [{} for _ in range(a.np)]
            seeds = [1729 + 100 * a.boot + 1000 * di + 10 * rep + s for s in range(a.np)]
            ths = [threading.Thread(target=stream, args=(url, prompts[(s, d)], s, seeds[s], a.n_predict, outs[s]))
                   for s in range(a.np)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            m = round_metrics(outs)
            row["rounds"].append(m)
            save()
            print(json.dumps({"arm": a.arm, "boot": a.boot, "depth": d, "rep": rep,
                              "agg": round(m.get("agg_tg") or 0, 1),
                              "tg": [round(p["tg"] or 0, 1) for p in m.get("streams", [])],
                              "acc": f'{m.get("draft_accepted")}/{m.get("draft_n")}',
                              "overlap": round(m.get("overlap_frac") or 0, 2),
                              "prefill_s": [round(p["prompt_ms"] / 1000, 1) if p["prompt_ms"] else None
                                            for p in row["prefill"]],
                              "err": m.get("error")}), flush=True)
            if m.get("checkpoint_miss"):
                row["note"] = "checkpoint miss: timed request re-prefilled; further reps at this depth skipped"
                break
        if res.get("stopped"):
            break
    else:
        res["complete"] = True
    save()


# ---------------------------------------------------------------- summary

def mean_sem(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None), len(xs)


def fmt(xs, nd=1):
    m, s, n = mean_sem(xs)
    if m is None:
        return "—"
    return f"{m:.{nd}f} ± {s:.{nd}f}" if s is not None else f"{m:.{nd}f} (n=1)"


def boot_sample(row):
    """One boot's numbers at one depth: rounds averaged, acceptance pooled."""
    ok = [r for r in row["rounds"] if not r.get("error") and not r.get("checkpoint_miss")]
    if not ok:
        return None
    streams = [p for r in ok for p in r["streams"]]
    pre = [p for p in row["prefill"] if p.get("prompt_ms")]
    dn = sum(r["draft_n"] for r in ok)
    return {"agg": statistics.fmean(r["agg_tg"] for r in ok if r["agg_tg"]),
            "stream": statistics.fmean(p["tg"] for p in streams if p["tg"]),
            "server": statistics.fmean(p["server_tg"] for p in streams if p["server_tg"]),
            "acc": sum(r["draft_accepted"] for r in ok) / dn if dn else None,
            "pp": statistics.fmean(p["prompt_per_second"] for p in pre) if pre else None,
            "pp_s": statistics.fmean(p["prompt_ms"] / 1000 for p in pre) if pre else None,
            "pp_n": statistics.fmean(p["prompt_n"] for p in pre) if pre else None,
            "overlap": min(r["overlap_frac"] for r in ok), "rounds": len(ok),
            "think_closed": sum(p["think_closed"] for p in streams), "n_streams": len(streams)}


def smi_min_free(path):
    free = {}
    try:
        for line in open(path):
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 3 and f[1].isdigit():
                free[f[1]] = min(free.get(f[1], 1e9), float(f[3]))
    except (OSError, ValueError, IndexError):
        pass
    return {k: int(v) for k, v in sorted(free.items())}


def summary(a):
    out = Path(a.dir)
    boots = []
    for f in sorted(glob.glob(str(out / "raw" / "*.json"))):
        if f.endswith(".launch.json"):
            continue
        try:
            boots.append(json.loads(Path(f).read_text()) | {"_file": f})
        except ValueError:
            continue
    arms = sorted({b["arm"] for b in boots}, key=lambda x: (x.split("-")[0], x))
    depths = sorted({r["depth"] for b in boots for r in b["results"]})
    L = ["# llama-agg: Swift 1.5 IQ4_XS aggregate decode, -np 1 vs -np 2 (queue item 30)", ""]
    fit = out / "vram_fit.txt"
    L += [f"Generated {time.strftime('%F %T')}. One boot = one sample; cells are mean ± SEM over boots (n in col 3). "
          "Arm P: `--spec-draft-sampling probabilistic --no-spec-draft-backend-sampling`, q8_0 target KV, f16 draft KV, "
          "`-sm tensor 50,50 -b 2048 -ub 128 --cache-ram 0`, temp 1 / top-k 20 / top-p 0.95 / min-p 0.05, reasoning on "
          "(chat template, default effort). Prompts: frozen corpus, nested depths per slot; decode "
          f"{boots[0]['n_predict'] if boots else '?'} tokens with ignore_eos from a cache hit.", "",
          "agg = all streams' tokens in the overlap window / window; stream = one stream's client-side tok/s; "
          "server = llama-server `predicted_per_second`; acc = accepted/drafted (pooled); prefill = prefill of the "
          "depth increment with the other slot idle (8k cold, 32k and 64k from the previous depth's checkpoint).", ""]
    L += ["| arm | depth | boots | agg tok/s | stream tok/s | server tok/s | acceptance | prefill tok/s | "
          "prefill s (tokens) |", "|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for arm in arms:
        for d in depths:
            ss = [boot_sample(r) for b in boots if b["arm"] == arm for r in b["results"] if r["depth"] == d]
            ss = [s for s in ss if s]
            if not ss:
                continue
            table[(arm, d)] = ss
            pn = statistics.fmean(s["pp_n"] for s in ss if s["pp_n"]) if any(s["pp_n"] for s in ss) else None
            L.append(f"| {arm} | {d} | {len(ss)} | **{fmt([s['agg'] for s in ss])}** | {fmt([s['stream'] for s in ss])} "
                     f"| {fmt([s['server'] for s in ss])} | {fmt([s['acc'] for s in ss], 3)} | "
                     f"{fmt([s['pp'] for s in ss], 0)} | {fmt([s['pp_s'] for s in ss])} ({pn:.0f}) |"
                     if pn else f"| {arm} | {d} | {len(ss)} | **{fmt([s['agg'] for s in ss])}** | — | — | — | — | — |")
    L += ["", "## -np 2 vs -np 1 (aggregate, same k)", "", "| depth | k | np1 agg | np2 agg | ratio |", "|---|---|---|---|---|"]
    for d in depths:
        for k in (2, 3, 4):
            one, two = table.get((f"np1-k{k}", d)), table.get((f"np2-k{k}", d))
            if one and two:
                m1, m2 = mean_sem([s["agg"] for s in one])[0], mean_sem([s["agg"] for s in two])[0]
                L.append(f"| {d} | {k} | {m1:.1f} | {m2:.1f} | {m2 / m1:.2f}× |")
    L += ["", "## Best arm per depth (highest mean aggregate; check SEM before calling a winner)", ""]
    for d in depths:
        cand = [(mean_sem([s["agg"] for s in v])[0], arm) for (arm, dd), v in table.items() if dd == d]
        if cand:
            m, arm = max(cand)
            L.append(f"- {d}: {arm} {m:.1f} tok/s")
    L += ["", "## Boots", "", "| file | arm | round | ctx | n_ctx_slot | load s | complete | min free MiB (GPU0/GPU1) | notes |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b in boots:
        la = b.get("launch", {})
        free = smi_min_free(b["_file"].replace(".json", ".smi.csv"))
        notes = [b.get("stopped", "")] + [r.get("note", "") for r in b["results"]]
        low_overlap = [r["depth"] for r in b["results"] for x in r["rounds"]
                       if b["np"] > 1 and (x.get("overlap_frac") or 1) < 0.8]
        if low_overlap:
            notes.append(f"overlap < 0.8 at {sorted(set(low_overlap))}")
        errs = [x["error"] for r in b["results"] for x in r["rounds"] if x.get("error")]
        if errs:
            notes.append(f"{len(errs)} round errors: {errs[0][:80]}")
        L.append(f"| {Path(b['_file']).name} | {b['arm']} | {b['round']} | {b['ctx']} | {la.get('n_ctx_slot', '?')} | "
                 f"{la.get('load_s', '?')} | {b['complete']} | {'/'.join(str(v) for v in free.values()) or '?'} | "
                 f"{'; '.join(n for n in notes if n)} |")
    if fit.exists():
        L += ["", "## VRAM fit (-np 2)", "", "```", fit.read_text().strip(), "```"]
    closed = sum(s["think_closed"] for v in table.values() for s in v)
    total = sum(s["n_streams"] for v in table.values() for s in v)
    L += ["", f"Reasoning check: {closed}/{total} timed streams closed `</think>` within the decode budget "
          "(ignore_eos keeps generating after an answer; a high share would mean the tail is not agent-like text).", ""]
    shas = {}
    for b in boots:
        for key, h in b.get("prompt_info", {}).get("sha256", {}).items():
            shas.setdefault(key, set()).add(h)
    drift = [k for k, v in shas.items() if len(v) > 1]
    L.append(f"Prompt identity across boots: {'OK' if not drift else 'DIFFERENT for ' + ', '.join(drift)} "
             f"({len(shas)} prompts).")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("boot")
    b.add_argument("--port", type=int, required=True)
    b.add_argument("--arm", required=True)
    b.add_argument("--np", type=int, required=True)
    b.add_argument("--k", type=int, required=True)
    b.add_argument("--ctx", type=int, required=True)
    b.add_argument("--boot", type=int, required=True)
    b.add_argument("--round", type=int, required=True)
    b.add_argument("--depths", default="8192,32768,65536")
    b.add_argument("--reps", type=int, default=2)
    b.add_argument("--n-predict", type=int, default=512)
    b.add_argument("--deadline", type=float, required=True, help="epoch s; no new depth or round after it")
    b.add_argument("--meta", help="launch.json written by the item")
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summary")
    s.add_argument("dir")
    a = ap.parse_args()
    boot(a) if a.cmd == "boot" else summary(a)


if __name__ == "__main__":
    main()
