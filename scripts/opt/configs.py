"""Engine configuration identity from a recorded launch command.

Contract: config_id = sha256 of the canonical TOML of `canon`. Volatile values (ports,
scratch paths, device UUIDs) never enter `canon`; paths are replaced by hashes when the
launch record has them. Everything non-identifying stays out of the id.
"""
import hashlib
import json
import re

from .store import canonical_toml, config_id_of

VOLATILE_FLAGS = {"host", "port", "alias", "metrics", "served-model-name", "sse-keep-alive-interval",
                  "chat-template-file", "chat-template", "m", "model", "jinja", "api-server-count",
                  "log-file", "cache-ram"}
ENV_PREFIXES = ("GGML_", "LLAMA_", "VLLM_", "NCCL_", "PYTORCH_", "CUDA_DEVICE_ORDER")
ENV_DROP = {"VLLM_HOST_STAGED_AR_LIB", "VLLM_SERVER_DEV_MODE", "LLAMA_ROUND_TRACE"}
QUANT_PATTERNS = [
    (r"IQ4[_-]?XS", "IQ4_XS"), (r"Q5[_-]?K[_-]?S", "Q5_K_S"), (r"Q5[_-]?K[_-]?M", "Q5_K_M"),
    (r"Q6[_-]?K", "Q6_K"), (r"IQ3[_-]?XXS", "IQ3_XXS"),
    (r"(?<![A-Za-z0-9])(J[1-9][a-z0-9]*)(?![A-Za-z0-9])", None),
]
HQ_CKPT = [("codexh", "hq:codexh"), ("attn-int8", "hq:g32i8"), ("syv-fast", "hq:g128")]
_SWIFT = re.compile(r"Swift(?:-(\d+(?:\.\d+)*))?-Qwen3\.8-27B-(.+)$")
# Flags already represented by a typed candidate field; everything else non-volatile is variant identity.
KEYED_FLAGS = {"np", "c", "ctx-size", "ctk", "ctv", "spec-draft-n-max", "spec-draft-p-min", "ub", "max-num-seqs",
               "max-model-len", "kv-cache-dtype", "speculative-config", "gpu-memory-utilization"}


def model_identity(path):
    """(model_id, family, version, recipe). Distinct recipes and base versions never share an id:
    swift-1.5:IQ4_XS vs swift-1.5:IQ4_XS-ssmout6, swift-1.0:W4A16-syv-fast vs swift-1.5:W4A16-syv-fast.
    Unrecognised names keep their whole stem so they can never collide with a Swift model."""
    stem = (path or "").rsplit("/", 1)[-1]
    stem = re.sub(r"\.gguf$", "", stem)
    m = _SWIFT.match(stem)
    if m:
        ver = m.group(1) or "1.0"
        return f"swift-{ver}:{m.group(2)}", "swift", ver, m.group(2)
    return f"other:{stem}", "other", None, stem


def variant_of(flags, env):
    """Non-volatile A/B identity beyond the typed candidate fields: other flags and GGML_/VLLM_/... env."""
    v = {"flags": {k: x for k, x in flags.items() if k not in KEYED_FLAGS}, "env": dict(env)}
    return v, hashlib.sha256(canonical_toml({"flags": v["flags"], "env": v["env"]}).encode()).hexdigest()[:8]


def quant_label(path):
    base = (path or "").rsplit("/", 1)[-1]
    if "W4A16" in base:
        for key, label in HQ_CKPT:
            if key in base:
                return label
        return "hq:" + base
    for pat, label in QUANT_PATTERNS:
        m = re.search(pat, base)
        if m:
            return label or m.group(1)
    return base.rsplit(".", 1)[0] or None


def build_family(binary):
    if not binary:
        return None
    p = binary.split("scratchpad/", 1)[-1]
    p = re.sub(r"^queue/build/", "", p)
    p = re.sub(r"/(build/)?bin(-[a-z]+)?/[^/]+$", "", p)
    p = re.sub(r"^/home/[^/]+/repos/", "", p)
    return p.replace("qwen38-weekend/frozen/", "")


_NEG_NUM = re.compile(r"-\d+(\.\d+)?")


def parse_flags(argv):
    """argv without program/model positionals -> {name: value|True}; later duplicates win."""
    flags, i = {}, 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-") and not _NEG_NUM.fullmatch(tok):
            name = tok.lstrip("-")
            if "=" in name:
                name, val = name.split("=", 1)
                flags[name] = val
            elif i + 1 < len(argv) and (not argv[i + 1].startswith("-") or _NEG_NUM.fullmatch(argv[i + 1])):
                flags[name] = argv[i + 1]
                i += 1
            else:
                flags[name] = True
        i += 1
    return flags


def _norm(v):
    if isinstance(v, str):
        if v[:1] in "{[":
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        if re.fullmatch(r"-?\d+\.\d+", v):
            return float(v)
        if "/" in v and not re.fullmatch(r"[\d.,/]+", v):
            return v.rsplit("/", 1)[-1]
    return v


def _clean_env(env):
    return {k: _norm(v) for k, v in (env or {}).items() if k.startswith(ENV_PREFIXES) and k not in ENV_DROP}


def from_command(command, env=None, *, build_sha=None, binary_sha256=None, model_sha256=None,
                 model_path=None, template_sha256=None, stack=None, vllm_version=None, extra_flags=None):
    """command: recorded argv (llama-server or `vllm serve`). Returns a config record."""
    command = list(command)
    is_vllm = "serve" in command[:4] and any(c.endswith("vllm") for c in command[:3])
    if is_vllm:
        s = command.index("serve")
        model_path = model_path or command[s + 1]
        raw = parse_flags(command[s + 2:])
        engine = "hq"
    else:
        raw = parse_flags(command[1:])
        model_path = model_path or raw.get("m") or raw.get("model")
        engine = "llama"
    if extra_flags:
        raw.update(parse_flags(extra_flags.split() if isinstance(extra_flags, str) else list(extra_flags)))
    flags = {k: _norm(v) for k, v in raw.items() if k not in VOLATILE_FLAGS and v is not None and v is not False}
    model_id, family, version, recipe = model_identity(model_path)
    model = {"label": quant_label(model_path), "id": model_id}
    model["sha256" if model_sha256 else "path"] = model_sha256 or model_path
    canon = {"engine": engine, "model": model, "flags": flags, "env": _clean_env(env)}
    if stack:
        canon["stack"] = stack
    build = {k: v for k, v in (("sha", build_sha), ("binary_sha256", binary_sha256),
                               ("vllm_version", vllm_version)) if v}
    if build:
        canon["build"] = build
    if template_sha256:
        canon["chat_template_sha256"] = template_sha256
    cid = config_id_of(canon)
    variant, variant_id = variant_of(flags, canon["env"])
    return {"id": cid, "config_id": cid, "engine": engine, "stack_name": stack,
            "build_sha": build_sha, "model_label": model["label"], "model_id": model_id, "model_family": family,
            "swift_version": version, "model_recipe": recipe, "model_path": model_path,
            "model_sha256": model_sha256, "variant": variant, "variant_id": variant_id,
            "canon": canon, "canonical_toml": canonical_toml(canon), **derive(engine, flags)}


def derive(engine, flags):
    d = {}
    if engine == "llama":
        d["np"] = flags.get("np", 1)
        d["ctx"] = flags.get("c") or flags.get("ctx-size")
        d["kv_k"], d["kv_v"] = flags.get("ctk", "f16"), flags.get("ctv", "f16")
        d["k"] = flags.get("spec-draft-n-max")
        d["p_min"] = flags.get("spec-draft-p-min")
        d["ub"] = flags.get("ub")
        if isinstance(d["ctx"], int) and isinstance(d["np"], int) and d["np"]:
            d["ctx_per_slot"] = d["ctx"] // d["np"]
    else:
        d["np"] = flags.get("max-num-seqs", 1)
        d["ctx"] = d["ctx_per_slot"] = flags.get("max-model-len")
        d["kv_k"] = d["kv_v"] = flags.get("kv-cache-dtype", "auto")
        spec = flags.get("speculative-config")
        d["k"] = spec.get("num_speculative_tokens") if isinstance(spec, dict) else None
        d["gpu_util"] = flags.get("gpu-memory-utilization")
    return {k: v for k, v in d.items() if v is not None}
