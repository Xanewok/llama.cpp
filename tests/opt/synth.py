"""Hand-built DB rows with known ground truth (no files)."""
from scripts.opt import configs
from scripts.opt.ingest.base import measurement
from scripts.opt.store import Store

PROV = {"out_dir": "synthetic", "summary_path": "synthetic/x.json", "summary_sha256": "0"}


def llama_cmd(quant="IQ4_XS", np_=1, ctx=102400, kv="q8_0", k=3, ub=128, split="1,1"):
    return ["/b/llama-server", "-m", f"/m/{quant}.gguf", "-np", str(np_), "-c", str(ctx), "-ctk", kv, "-ctv", kv,
            "--spec-draft-n-max", str(k), "-ub", str(ub), "--tensor-split", split, "-sm", "tensor"]


def add_cfg(store, cmd, stack="s", env=None, **kw):
    c = configs.from_command(cmd, env, stack=stack, **kw)
    store.upsert("configs", {**c, "id": c["config_id"]})
    return c


def add_m(store, cfg, metric, value, status="measured", source="campaign", workload="codex-8task-1agent", item="i", sha=None, **dims):
    m = measurement(source=source, metric=metric, value=value, unit="x", config_id=cfg["config_id"], workload_id=workload,
                    prov={**PROV, "summary_sha256": sha or item}, dims=dims, status=status, item_id=item, key=(item, str(value)))
    store.upsert("measurements", m)
    return m


def vram_points(store, quant, kv, k, np_, ctx, a0, a1, b, stack="s", c=100.0, ub=128):
    cfg = add_cfg(store, llama_cmd(quant, np_, ctx, kv, k, ub), stack=stack)
    for gpu, a in ((0, a0), (1, a1)):
        add_m(store, cfg, "vram_min_free_mib", a - b * ctx - c * k, gpu=gpu, np=np_, ctx=ctx)
    return cfg
