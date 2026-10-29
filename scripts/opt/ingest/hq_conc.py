"""hq-conc boots: raw/bootN/{fit.json, single-D.json, conc-D.json, prefix.json, serve.log}.

One unit = one boot directory. A failed prefix check is a measurement (prefix_kept = 0),
not a missing value: it is what bounds the two-stream context.
"""
import ast
import re
from pathlib import Path

from .. import configs
from ..store import Reject, sha256_file
from .base import Batch, load_json, measurement, need

DROP_ARGS = {"model_tag", "model", "chat_template", "host", "port", "served_model_name", "sse_keep_alive_interval",
             "enable_prompt_tokens_details", "compilation_config", "api_server_count"}


def _config_from_serve_log(log):
    text = Path(log).read_text(errors="replace") if Path(log).exists() else ""
    m = re.search(r"non-default args: (\{.*\})\s*$", text, re.M)
    if not m:
        raise Reject(f"{log}: no 'non-default args' line; cannot identify the HQ config")
    try:
        args = ast.literal_eval(re.sub(r"<[\w.]+: ('[^']*')>", r"\1", m.group(1)))
    except (ValueError, SyntaxError) as e:
        raise Reject(f"{log}: unparsable non-default args ({str(e)[:80]})")
    model = args.get("model") or args.get("model_tag")
    ver = re.search(r"version (\d+\.\d+\.\d+)", text)
    argv = ["python", "vllm", "serve", model]
    for k, v in args.items():
        if k in DROP_ARGS or v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is not False:
            argv += [flag, __import__("json").dumps(v) if isinstance(v, (dict, list)) else str(v)]
    return configs.from_command(argv, None, vllm_version=ver.group(1) if ver else None,
                                stack=configs.quant_label(model))


def parse(path):
    bdir = Path(path)
    fit_p = bdir / "fit.json"
    if not fit_p.exists():
        raise Reject(f"{bdir}: no fit.json (not an hq-conc boot directory)")
    fit = load_json(fit_p)
    need(fit, ["kv_pool_tokens"], str(fit_p))
    b = Batch()
    cfg = _config_from_serve_log(bdir / "serve.log")
    cid = b.add_config(cfg)
    parts = bdir.parts
    job = "/".join(parts[parts.index("queue-out") + 1:]) if "queue-out" in parts else bdir.name
    wl = "synthetic:hq-conc"
    prov = {"out_dir": str(bdir), "summary_path": str(fit_p), "summary_sha256": sha256_file(fit_p) + (sha256_file(bdir / "serve.log") if (bdir / "serve.log").exists() else ""),
            "model_sha256": None}
    common = dict(source="hq-conc", config_id=cid, workload_id=wl, prov=prov, item_id=job, attempt=bdir.name)

    def add(metric, value, unit, **kw):
        if value is not None:
            b.measurements.append(measurement(metric=metric, value=value, unit=unit, **{**common, **kw}))

    add("kv_pool_tokens", fit["kv_pool_tokens"], "tokens")
    for f in sorted(bdir.glob("single-*.json")):
        d = load_json(f)
        need(d, ["runs"], str(f))
        for r in d["runs"]:
            need(r, ["depth", "tg", "pp"], str(f))
            dims = {"depth": r["depth"]}
            add("decode_tok_s", r["tg"], "tok/s", dims=dims, n=1, key=(f.name,))
            add("prefill_tok_s", r["pp"], "tok/s", dims=dims, key=(f.name,))
            if r.get("drafted"):
                add("draft_acceptance", r["accepted"] / r["drafted"], "ratio", dims=dims, key=(f.name,))
    for f in sorted(bdir.glob("conc-*.json")):
        d = load_json(f)
        need(d, ["runs"], str(f))
        for r in d["runs"]:
            need(r, ["depth"], str(f))
            dims = {"depth": r["depth"], "streams": 2}
            valid = r["valid"] if "valid" in r else r.get("agg_tg") is not None  # schema 1 has no flag
            if valid:
                if r.get("agg_tg") is None:
                    raise Reject(f"{f}: valid run without agg_tg")
                add("agg_tok_s", r["agg_tg"], "tok/s", dims=dims, n=2, key=(f.name,))
                for i, p in enumerate(r.get("per", [])):
                    if isinstance(p, dict) and p.get("tg") is not None:
                        add("stream_tok_s", p["tg"], "tok/s", dims={**dims, "stream": i}, key=(f.name,))
            else:
                reasons = "; ".join(r.get("invalid_reasons", [])) or "invalid"
                m = re.search(r"cached=(\d+), need >= (\d+)", reasons)
                if m:
                    add("prefix_kept_tokens", int(m.group(1)), "tokens", dims=dims, key=(f.name,),
                        reason=f"need >= {m.group(2)}")
                    add("prefix_kept", 0, "bool", dims=dims, key=(f.name,), reason=reasons)
                else:
                    add("conc_valid", 0, "bool", dims=dims, status="invalid", reason=reasons, key=(f.name,))
    pf = bdir / "prefix.json"
    if pf.exists():
        d = load_json(pf)
        need(d, ["valid", "cells", "pool_tokens_M"], str(pf))
        status, reason = ("measured", None) if d["valid"] and not d.get("canary") else (
            "invalid", "canary run" if d.get("canary") else "prefix.json valid=false")
        prov2 = {**prov, "summary_path": str(pf)}
        add("kv_pool_tokens", d["pool_tokens_M"], "tokens", key=("prefix",), prov=prov2)
        for c in d["cells"]:
            need(c, ["depth", "probes"], str(pf))
            kept = [bool(p["retained"]) for p in c["probes"]]
            if not kept:
                raise Reject(f"{pf}: cell depth={c['depth']} has no probes")
            dims = {"depth": c["depth"], "streams": len(kept)}
            add("prefix_kept_frac", sum(kept) / len(kept), "ratio", dims=dims, n=len(kept), status=status, reason=reason,
                key=("prefix",), prov=prov2)
            add("prefix_kept", int(all(kept)), "bool", dims=dims, n=len(kept), status=status, reason=reason,
                key=("prefix",), prov=prov2)
    return b
