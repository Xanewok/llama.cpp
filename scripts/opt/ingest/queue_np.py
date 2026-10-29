"""gpuq/queue steady-decode runs: <out>/{raw,cap}/NN-<arm>-rR.{json,launch.json,smi.csv}.

The summary.md files next to them are rendered from these files and are not parsed.
round_ms = (n / server_tg)*1000/(n - accepted drafts) per stream: the server's decode clock, not the client's
decode_s, exactly as depth.py (queue item 80) defines it for the summary.md files (one verify round emits
n/(n - accepted) tokens). Streams without server_tg are skipped, as there.
"""
import csv
import re
from pathlib import Path
from statistics import mean

from .. import configs
from ..store import Reject, sha256_file
from .base import Batch, load_json, measurement, need


def _min_free(smi):
    free = {}
    for i, row in enumerate(csv.reader(open(smi))):
        if len(row) < 4:
            raise Reject(f"{smi}:{i + 1}: malformed nvidia-smi row")
        try:
            free[int(row[1])] = min(free.get(int(row[1]), 1e18), float(row[3]))
        except ValueError:
            raise Reject(f"{smi}:{i + 1}: non-numeric memory.free")
    if not free:
        raise Reject(f"{smi}: empty")
    return free


def parse(path):
    jf = Path(path)
    if jf.name.endswith(".launch.json"):
        raise Reject(f"{jf}: pass the run json, not its launch record")
    run = load_json(jf)
    need(run, ["arm", "np", "k", "ctx", "complete", "results"], str(jf))
    launch_p = jf.with_name(jf.stem + ".launch.json")
    launch = load_json(launch_p)
    need(launch, ["command"], str(launch_p))
    src = launch.get("source") or ""
    m = re.match(r"([0-9a-f]{40})\s+(\S+)", src)
    sha = m.group(1) if m else None
    b = Batch()
    cfg = configs.from_command(launch["command"], launch.get("env"), build_sha=sha,
                               stack=configs.build_family(launch["command"][0]))
    cid = b.add_config(cfg)
    if cfg["np"] != run["np"] or cfg["ctx"] != run["ctx"]:
        raise Reject(f"{jf}: run np/ctx ({run['np']}/{run['ctx']}) disagree with launch record ({cfg['np']}/{cfg['ctx']})")
    out_dir = jf.parent.parent
    status, reason = ("measured", None) if run["complete"] else ("invalid", "run incomplete")
    prov = {"out_dir": str(out_dir), "summary_path": str(jf), "summary_sha256": sha256_file(jf),
            "build_sha": sha, "model_sha256": None}
    common = dict(source="queue-np", config_id=cid, workload_id="synthetic:steady-decode", prov=prov,
                  status=status, reason=reason, item_id=out_dir.name, attempt=jf.stem,
                  pair_group=f"{out_dir.name}:round{run.get('round')}", key=(jf.stem,))

    def add(metric, value, unit, **kw):
        if value is not None:
            b.measurements.append(measurement(metric=metric, value=value, unit=unit, **{**common, **kw}))

    np_ = run["np"]
    for res in run["results"]:
        d = res["depth"]
        ok = [r for r in res["rounds"] if not r.get("error") and not r.get("checkpoint_miss")]
        streams = [s for r in ok for s in r["streams"] if s.get("n")]
        if not streams:
            b.flag(jf, f"depth {d}: no error-free, checkpoint-hit rounds; skipped")
            continue
        live = [s for s in streams if s.get("server_tg") and s["n"] > s["draft_accepted"]]
        rms = [s["n"] / s["server_tg"] * 1000 / (s["n"] - s["draft_accepted"]) for s in live]
        tokr = [s["n"] / (s["n"] - s["draft_accepted"]) for s in live]
        dims = {"depth": d, "np": np_}
        spread = (max(rms) - min(rms)) if len(rms) > 1 else None
        if rms:
            add("round_ms", mean(rms), "ms", dims=dims, n=len(rms), spread=spread)
            add("acc_tok_per_round", mean(tokr), "tok", dims=dims, n=len(tokr))
        tgs = [s["server_tg"] for s in streams if s.get("server_tg")]
        if tgs:
            add("decode_tok_s", mean(tgs), "tok/s", dims=dims, n=len(streams))
        drafted = sum(s["draft_n"] for s in streams)
        if drafted:
            add("draft_acceptance", sum(s["draft_accepted"] for s in streams) / drafted, "ratio", dims=dims, n=len(streams))
        cold = [p["prompt_per_second"] for p in res.get("prefill", []) if p.get("cache_n", 0) == 0 and p.get("prompt_per_second")]
        if cold:
            add("prefill_tok_s", mean(cold), "tok/s", dims=dims, n=len(cold))
        aggs = [r["agg_tg"] for r in ok if r.get("agg_tg") and len(r["streams"]) == np_ and np_ > 1]
        if aggs:
            add("agg_tok_s", mean(aggs), "tok/s", dims={**dims, "streams": np_}, n=len(aggs),
                spread=(max(aggs) - min(aggs)) if len(aggs) > 1 else None)
        miss = sum(bool(r.get("checkpoint_miss")) for r in res["rounds"])
        if miss:
            add("checkpoint_miss_rounds", miss, "count", dims=dims)
    smi = jf.with_suffix(".smi.csv")
    if smi.exists():
        for gpu, v in sorted(_min_free(smi).items()):
            add("vram_min_free_mib", v, "MiB", dims={"gpu": gpu, "np": np_, "ctx": run["ctx"]})
    else:
        b.flag(jf, "no smi.csv: VRAM min-free not recorded")
    return b
