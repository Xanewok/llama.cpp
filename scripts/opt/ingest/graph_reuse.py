"""graph-reuse A/B: ab-summary.json (+ per-boot command.json/SOURCE for the config).

Field suffixes follow the producer's convention: *_M = measured, *_D = derived from
measured values; they map straight onto the row status.
"""
import re
from pathlib import Path

from .. import configs
from ..store import Reject, sha256_file
from .base import Batch, load_json, measurement, need

METRICS = {
    "steady_ms_per_round_D": ("steady_round_ms", "ms"),
    "whole_request_ms_per_round_D": ("whole_request_ms_per_round", "ms"),
    "request_fixed_ms_estimate_D": ("request_fixed_ms", "ms"),
    "prompt_ms_M": ("prompt_ms", "ms"),
    "wall_ms_M": ("request_wall_ms", "ms"),
    "generation_excess_ms_D": ("generation_excess_ms", "ms"),
}


def _status(src_key):
    return "measured" if src_key.endswith("_M") else "derived"


def _boot_config(run, boot, arm, k_label, models):
    d = run / ((f"k{k_label}-" if k_label else "") + f"{boot}-{arm}")
    cmdf = d / "command.json"
    if not cmdf.exists():
        raise Reject(f"{d}: command.json missing; cannot identify the config of arm {arm}")
    c = load_json(cmdf)
    need(c, ["command"], str(cmdf))
    cmd = list(c["command"])
    src = (d / "SOURCE") if (d / "SOURCE").exists() else (d / "bin" / "SOURCE")
    sha = src.read_text().split()[0] if src.exists() else None
    sums = d / "bin" / "SHA256SUMS"
    bsha = None
    if sums.exists():
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == Path(cmd[0]).name:
                bsha = parts[0]
    mpath = cmd[cmd.index("-m") + 1] if "-m" in cmd else None
    return configs.from_command(cmd, None, build_sha=sha, binary_sha256=bsha,
                                model_sha256=models.get(mpath), stack=f"graph-reuse:{arm}")


def parse(path):
    sp = Path(path)
    data = load_json(sp)
    run = sp.parent
    blocks = data if isinstance(data, list) else [data]
    inputs = run / "inputs.json"
    models = {k: v for k, v in (load_json(inputs).items() if inputs.exists() else []) if k.endswith(".gguf")}
    ssha = sha256_file(sp)
    out = Batch()
    for blk in blocks:
        need(blk, ["requests", "k"], str(sp))
        k = blk["k"]
        tandem = isinstance(data, list)
        cfgs = {}
        for r in blk["requests"]:
            need(r, ["boot", "arm", "case"], str(sp))
            key = (r["boot"], r["arm"])
            if key not in cfgs:
                cfgs[key] = out.add_config(_boot_config(run, r["boot"], r["arm"], k if tandem else None, models))
            cid = cfgs[key]
            prov = {"out_dir": str(run), "summary_path": str(sp), "summary_sha256": ssha,
                    "build_sha": out.configs[cid].get("build_sha"), "model_sha256": out.configs[cid].get("model_sha256")}
            for src_key, (metric, unit) in METRICS.items():
                v = r.get(src_key)
                if v is None:
                    continue
                if not isinstance(v, (int, float)):
                    raise Reject(f"{sp}: non-numeric {src_key}={v!r}")
                out.measurements.append(measurement(
                    source="graph-reuse-ab", metric=metric, value=v, unit=unit, config_id=cid,
                    workload_id=f"synthetic:graph-reuse-ab:k{k}", dims={"case": r["case"], "arm": r["arm"], "boot": r["boot"]},
                    status=_status(src_key), prov=prov, item_id=run.name, pair_group=f"{run.name}:k{k}:{r['case']}",
                    key=(k,)))
        up = next((cfgs[kk] for kk in cfgs if kk[1] == "up"), None)
        for pr in blk.get("paired", []):
            arm = pr.get("arm", "patch")
            cid = next((cfgs[kk] for kk in cfgs if kk[1] == arm), None)
            if not cid or not up or "patch_minus_up_D" not in pr:
                continue
            for src_key, (metric, unit) in METRICS.items():
                v = pr["patch_minus_up_D"].get(src_key)
                if v is not None:
                    out.measurements.append(measurement(
                        source="graph-reuse-ab", metric="delta_" + metric, value=v, unit=unit, config_id=cid,
                        workload_id=f"synthetic:graph-reuse-ab:k{k}", dims={"case": pr["case"], "vs_config": up, "arm": arm},
                        n=len(pr.get("pair_differences_D", [])) or 1, status="derived",
                        prov={"out_dir": str(run), "summary_path": str(sp), "summary_sha256": ssha},
                        item_id=run.name, pair_group=f"{run.name}:k{k}:{pr['case']}", key=(k,)))
    if not out.measurements:
        raise Reject(f"{sp}: no request rows")
    return out
