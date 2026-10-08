"""harnesslib: what makes a number a measurement, shared by every harness kind.

gpuq owns the queue gates (Xid verdict, idle GPUs, RAM, ports, memoc, wall clock, pins); nothing here re-decides them.
"""
import csv
import functools
import hashlib
import io
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import time
import threading
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from . import calibration

EXIT = {3: "hazard", 4: "transient", 5: "insufficient_data", 6: "integrity", 7: "timeout", 8: "deterministic"}
RC = {cls: rc for rc, cls in EXIT.items()}
XID_RE = r"\bXid\b|fallen off the bus"
# Fallback for an unmapped rc only (gpuq applies exit_codes first). Every alternative is a phrase or word-bounded.
SIGNATURES = [
    ("hazard", XID_RE),
    ("integrity", r"frozen bytes changed|inputs changed|model SHA changed|effective params mismatch|frozen file set changed|"
                  r"cross-arm identity failed|path-engaged proof failed|staging-engaged proof failed|canary prompt differs"),
    ("insufficient_data", r"too few verify steps|proof inconclusive|missing final token identity|canary prompt has \d+ tokens|"
                          r"reached only \d+ cached|R2 reused only"),
]
SERVER_NAME = "llama-server"  # gpuq's stop hook and desktop gate key on it; an arm's executable must never carry it
FIT_JOB = "agentic-fit"  # hillclimb job-id prefix of the capacity probe: it runs without the agentic sampling flags
PRE_READY = {"startup_timeout", "server_failure", "oom", "vram_fit_unavailable"}  # hillclimb reasons before /props
MODEL_OUTCOMES = {"probe_failed", "probe_incomplete"}  # model behaviour, kept as data


class Fail(ValueError):
    def __init__(self, message, cls="deterministic"):
        super().__init__(message)
        self.cls, self.rc = cls, RC[cls]


def log(out, message):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        with (Path(out) / "console.log").open("a") as stream:
            stream.write(line + "\n")


def event(text):
    path = os.environ.get("GPUQ_EVENTS")
    if path:
        with open(path, "a") as stream:
            stream.write(text + "\n")


def write_json(path, value):
    """Atomic: a reader never sees half a record. NaN and inf are refused, not written."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worst_class(classes):
    order = ("hazard", "integrity", "timeout", "insufficient_data", "transient", "deterministic")
    return min(classes, key=order.index)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------- experiment

def _expand(value, ctx):
    if isinstance(value, str):
        for key, sub in ctx.items():
            value = value.replace("{" + key + "}", str(sub))
    return value


REQUEST_KEYS = ("seed", "n_predict", "temperature", "top_k", "top_p", "min_p", "ignore_eos")


def read_experiment(path):
    """The experiment file is the only place experiment facts live; this resolves paths and refuses an incomplete one."""
    path = Path(path).resolve()
    source = path.read_bytes()
    x = tomllib.loads(source.decode())
    x["file"] = path
    x["_file_sha"] = hashlib.sha256(source).hexdigest()

    def need(ok, what):
        if not ok:
            raise Fail(f"inputs changed: {path.name}: {what}", "integrity")

    need(re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", x.get("id", "")), "id")
    need("priority" not in x or type(x["priority"]) is int, "priority must be an integer")
    need(x.get("kind") in ("campaign", "depth", "probe"), "kind must be campaign, depth or probe")
    x["plan"] = list(x.get("plan", []))
    arms = x.setdefault("arms", {})
    need(arms and set(x["plan"]) <= set(arms), "plan names an undefined arm")
    for name, arm in arms.items():
        arm["name"] = name
        arm["root"] = Path(arm["root"])
        arm["runroot"] = Path(arm["runroot"]) if arm.get("runroot") else None
        arm["repo"] = Path(arm["repo"]) if arm.get("repo") else None
        ctx = {"root": arm["root"], "runroot": arm["runroot"] or ""}
        arm["exe"] = arm["root"] / arm["exe"]
        arm["bindir"] = arm["root"] / arm.get("bin", "bin")  # `bin` names another layout (e.g. build/bin), which has no top-level SHA256SUMS
        need(arm["exe"].name != SERVER_NAME, f"arm {name} executable is named {SERVER_NAME}")
        arm["env"] = {k: _expand(v, ctx) for k, v in arm.get("env", {}).items()}
        arm.setdefault("flags", [])
        arm.setdefault("pinned", {})
        arm.setdefault("canary_log", [])
        need(re.fullmatch(r"[0-9a-f]{40}", arm.get("commit", "")), f"arm {name} commit")
    for key in ("model", "draft_vocab"):
        if key in x:
            x[key]["path"] = Path(x[key]["path"])
    d = x.get("depth", {})
    need(isinstance(d.get("streams", 1), int) and 1 <= d.get("streams", 1) <= 8, "depth.streams must be an integer in 1..8")
    need(0 < d.get("min_overlap", 0.8) <= 1, "depth.min_overlap must be in (0, 1]")
    need(d.get("decode_start", "joint") in ("threads", "joint"), 'depth.decode_start must be "threads" or "joint"')
    for what, n in (("depth.texts", d.get("texts", 1)), ("canary.texts", x.get("canary", {}).get("texts", 1))):
        need(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 64, f"{what} must be an integer in 1..64")
        need(n == 1 or d.get("streams", 1) == 1, f"{what} > 1 needs depth.streams = 1")
    need(isinstance(x.get("campaign", {}).get("agents", 1), int) and 1 <= x.get("campaign", {}).get("agents", 1) <= 8, "campaign.agents must be an integer in 1..8")
    request_keys = set(REQUEST_KEYS)
    need(set(d.get("request", {})) <= request_keys, f"depth.request keys {sorted(set(d.get('request', {})) - request_keys)} are not request keys {sorted(request_keys)}")
    for name, arm in arms.items():
        need("request" not in arm or x["kind"] == "depth", f"arm {name}: request overrides exist for depth experiments only")
        need(set(arm.get("request", {})) <= request_keys, f"arm {name} request keys {sorted(set(arm.get('request', {})) - request_keys)} are not request keys {sorted(request_keys)}")
    need(all(isinstance(r.get("ignore_eos", True), bool) for r in [d.get("request", {})] + [a.get("request", {}) for a in arms.values()]), "request.ignore_eos must be a boolean")
    need(set(d.get("expected", {})) <= {str(v) for v in d.get("depths", [])}, "depth.expected names a depth outside depth.depths")
    need(all(isinstance(k, str) for k in d.get("counters", [])), "depth.counters must be round-trace counter kinds")
    if "cpu_evidence" in x:
        need(set(x["cpu_evidence"]) == {"path", "command"} and Path(x["cpu_evidence"]["path"]).is_absolute(), "cpu_evidence = {path, command}, path absolute")
    q = x.get("question", {})
    need(isinstance(q, dict) and set(q) <= {"expected", "falsifier", "answers"} and all(isinstance(v, str) and v.strip() for v in q.values()),
         "[question] takes non-empty strings expected, falsifier, answers")
    if "expected" in q:
        try:
            calibration.parse_expected(q["expected"])
        except ValueError as error:
            need(False, f"question.{error}")
    x.setdefault("server", {})
    _cache_ram(x, need)
    x.setdefault("canary", {})
    x.setdefault("budget", {})
    return x


CACHE_RAM_FLAGS = ("--cache-ram", "-cram")


def _cache_ram(x, need):
    """The host prompt cache (default 8 GiB, 3-8 GiB per entry) swaps the host and is never used by timed decode, so every
    llama-server runs `--cache-ram 0`. Contract: [server] cache_ram (default "0") and arm.cache_ram override it, "default"
    omits the flag (serve.sh behaviour); a template that spells the flag itself is taken as written (no per-arm override)."""
    s, kind = x["server"], x["kind"]
    tokens = s.get("args", []) + s.get("agentic_args", []) + s.get("extra", [])
    literal = any(t in CACHE_RAM_FLAGS or t.startswith("--cache-ram=") for t in tokens)
    for name, arm in x["arms"].items():
        value = str(arm.get("cache_ram", s.get("cache_ram", "0")))
        need(value == "default" or value.isdigit(), f"arm {name}: cache_ram must be MiB or \"default\"")
        need(not (literal and "cache_ram" in arm), f"arm {name}: cache_ram needs a template without a literal --cache-ram")
        arm["cache_ram_flags"] = [] if value == "default" or literal else ["--cache-ram", value]
    part = "extra" if kind == "campaign" else "args"
    if not literal and (part in s or kind == "campaign"):
        s[part] = [*s.get(part, []), "{cache_ram_flags}"]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- frozen bytes

def sums_file(path):
    """Every line of a sha256sum manifest must hold for files inside its own directory; returns {path: digest}."""
    path = Path(path)
    base, files = path.parent.resolve(), {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name[1:] if name.startswith("*") else name
        target = path.parent / name
        if not target.resolve().is_relative_to(base):
            raise Fail(f"frozen bytes changed: {name} escapes {path}", "integrity")
        try:
            ok = file_sha(target) == digest
        except OSError:
            ok = False
        if not ok:
            raise Fail(f"frozen bytes changed ({path}): {name}", "integrity")
        files[str(target)] = digest
    return files


def check_frozen(arm):
    """The arm's bytes are the frozen ones: per-tree sums, pinned top-level sums, SOURCE commit, every bin file hashed, and
    a runroot that only links into the frozen bin. Returns {path: digest} of everything checked."""
    root, runroot, bindir = arm["root"], arm["runroot"], arm["bindir"]
    top = sums_file(root / "SHA256SUMS") if (root / "SHA256SUMS").exists() or "bin" not in arm else {}
    files = top | sums_file(bindir / "SHA256SUMS")
    top = {Path(p).relative_to(root).as_posix(): d for p, d in top.items()}
    for name, digest in arm["pinned"].items():
        if top.get(name) != digest:
            raise Fail(f"frozen bytes changed: {root}/SHA256SUMS {name} differs from the pinned hash", "integrity")
    if arm["pinned"] and set(top) != set(arm["pinned"]):
        raise Fail(f"frozen file set changed: {root}/SHA256SUMS lists {sorted(top)}", "integrity")
    hashed = {Path(p).name for p in files if Path(p).parent == bindir} - {"SHA256SUMS"}
    if {p.name for p in bindir.iterdir()} - {"SHA256SUMS"} != hashed:
        raise Fail(f"frozen file set changed: unhashed files in {bindir}", "integrity")
    exe = arm["exe"]
    if exe.name == SERVER_NAME or str(exe) not in files:
        raise Fail(f"frozen bytes changed: arm {arm['name']} executable {exe} is not a hashed frozen file", "integrity")
    source = (root / "SOURCE").read_text().split()[0]
    if source != arm["commit"]:
        raise Fail(f"frozen bytes changed: arm {arm['name']} SOURCE {source} != {arm['commit']}", "integrity")
    if arm["repo"]:
        seen = subprocess.run(["git", "-C", str(arm["repo"]), "rev-parse", f"{source}^{{commit}}"], capture_output=True, text=True).stdout.strip()
        if seen != arm["commit"]:
            raise Fail(f"frozen bytes changed: {arm['repo']} does not hold {arm['commit']}", "integrity")
    if runroot:
        files |= sums_file(runroot / "RUNROOT.SHA256SUMS")
        if (runroot / "SOURCE").read_text().split()[0] != arm["commit"]:
            raise Fail(f"frozen bytes changed: runroot SOURCE of arm {arm['name']}", "integrity")
        if (runroot / "build/bin" / SERVER_NAME).resolve() != exe.resolve():
            raise Fail(f"frozen bytes changed: runroot server resolves to {(runroot / 'build/bin' / SERVER_NAME).resolve()}", "integrity")
        for link in (runroot / "build/bin").iterdir():
            if not link.is_symlink() or bindir.resolve() not in link.resolve().parents:
                raise Fail(f"frozen bytes changed: runroot entry {link.name} is not a link into the frozen bin", "integrity")
    return files


def check_model(model, full=False):
    """Size always (cheap); the full digest only where it is affordable (CPU preflight). gpuq pins re-hash at start."""
    path = model["path"]
    if "bytes" in model and path.stat().st_size != model["bytes"]:
        raise Fail(f"model SHA changed: {path} size {path.stat().st_size} != {model['bytes']}", "integrity")
    if full and file_sha(path) != model["sha"]:
        raise Fail(f"model SHA changed: {path}", "integrity")


def check_input(spec, what):
    if file_sha(spec["path"]) != spec["sha"]:
        raise Fail(f"inputs changed: {what} {spec['path']}", "integrity")


# ---------------------------------------------------------------- launch contract

def expand_args(template, arm, x, port, more=None):
    """The experiment's argv template with {model} {draft_vocab} {port} {k} {p_min} {root} (+ `more`, e.g. {np} {ctx});
    {arm_flags} splices the arm's flags, {cache_ram_flags} its host-cache flag (see _cache_ram)."""
    ctx = {"model": x["model"]["path"], "port": port, "root": arm["root"], "k": arm.get("k", ""), "p_min": arm.get("p_min", "")} | (more or {})
    if "draft_vocab" in x:
        ctx["draft_vocab"] = x["draft_vocab"]["path"]
    out = []
    for token in template:
        if token in ("{arm_flags}", "{cache_ram_flags}"):
            out += list(arm["flags" if token == "{arm_flags}" else "cache_ram_flags"])
        else:
            out.append(_expand(token, ctx))
    return out


def pairs(argv):
    """(flag, value) for valued flags and (flag, None) for switches; a token after a flag is its value unless it is a flag."""
    out, i = [], 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            out.append((token, argv[i + 1]))
            i += 2
        else:
            out.append((token, None))
            i += 1
    return out


def flag_values(cmd, flag):
    return [cmd[i + 1] for i, token in enumerate(cmd[:-1]) if token == flag]


def verify_launch(cmd, contract, argv0, executed=None, exe=None, optional=(), allow=(), extra_ok=()):
    """Raise integrity unless argv is exactly the contract: every contract pair once with its value ({port}: any), every
    switch once, no flag outside contract + allow, optional flags all-or-nothing, executed = exe + the same args."""
    cmd = [str(c) for c in cmd]
    problems = []
    if cmd[0] != str(argv0):
        problems.append(f"argv[0] {cmd[0]} != {argv0}")
    args = list(cmd[1:])
    for flag, value in extra_ok:
        for i in range(len(args) - 1):
            if args[i] == flag and args[i + 1] == value:
                del args[i:i + 2]
                break
    want = pairs(contract)
    present = {f for f, _ in pairs(args)}
    opt = [f for f, _ in want if f in optional]
    if opt and any(f in present for f in opt) and not all(f in present for f in opt):
        problems.append(f"optional flags partly present: {[f for f in opt if f not in present]}")
    for flag, value in want:
        if flag in optional and flag not in present:
            continue
        got = flag_values(args, flag) if value is not None else None
        if value is None:
            if args.count(flag) != 1:
                problems.append(f"{flag} count {args.count(flag)}")
        elif value == "{port}":
            if len(got) != 1:
                problems.append(f"{flag} {got}")
        elif got != [value]:
            problems.append(f"{flag} {got} != [{value}]")
    known = {f for f, _ in want} | set(allow)
    problems += [f"unexpected {f}" for f in sorted(present - known)]
    if executed is not None:
        executed = [str(c) for c in executed]
        if exe is not None and executed[0] != str(exe):
            problems.append(f"executed binary {executed[0]} != {exe}")
        if Path(executed[0]).name == SERVER_NAME:
            problems.append(f"server named {SERVER_NAME}")
        if executed[1:] != cmd[1:]:
            problems.append("executed args differ from the recorded command")
    if problems:
        raise Fail("effective params mismatch (argv): " + "; ".join(problems), "integrity")


def check_shape(command, slots, ctx):
    """-np is the number of concurrent clients and -c their summed context; a literal that drifted from the experiment's
    clients x per-client context is not the experiment."""
    bad = [f"{flag} {flag_values(command, flag)} != [{value}]" for flag, value in (("-np", slots), ("-c", ctx))
           if flag_values(command, flag) != [str(value)]]
    if bad:
        raise Fail("effective params mismatch (argv): " + "; ".join(bad) + f" for {slots} concurrent clients", "integrity")


def server_env(base, arm, knobs, extra=None):
    """Scrubbed copy of base: no inherited knob or library path reaches the server; the arm's env (+ extra) on top."""
    env = {k: v for k, v in dict(base).items() if k not in knobs and k != "LD_LIBRARY_PATH"}
    return env | arm["env"] | (extra or {})


def env_record(env, knobs):
    return {k: env.get(k) for k in (*knobs, "LD_LIBRARY_PATH")}


def verify_env(record, want, knobs):
    record = record or {}
    problems = [f"{k}={record.get(k)!r} != {want.get(k)!r}" for k in (*knobs, "LD_LIBRARY_PATH") if record.get(k) != want.get(k)]
    if problems:
        raise Fail("effective params mismatch (env): " + "; ".join(problems), "integrity")


def verify_nice(nice, want=0):
    """The server inherits the launcher's nice; host time is a measured axis, so only CPU stages run niced."""
    if nice is None or nice > want:
        raise Fail(f"effective params mismatch (nice): server nice {nice} > {want}", "integrity")


def verify_props(props, ctx, slots=1):
    got = (props.get("total_slots"), props.get("default_generation_settings", {}).get("n_ctx"))
    if got != (slots, ctx):
        raise Fail(f"effective params mismatch (props): slots={got[0]} n_ctx={got[1]}", "integrity")


LOG_PARAMS = re.compile(r"n_max=(\d+),? n_min=(\d+),? p_min=([\d.]+)")
KV_TYPES = (re.compile(r"type_[kv]\s*=\s*(\w+)"), re.compile(r"\b[KV] \((\w+)\): +[\d.]+ MiB"))


def verify_server_log(text, k=None, p_min=None, kv_type=None):
    """Echoed speculative params must agree with the arm; every target KV type the log names must be kv_type. Returns echo count."""
    seen = LOG_PARAMS.findall(text)
    for n_max, n_min, p in seen:
        if k is not None and (int(n_max) != k or int(n_min) != 0 or abs(float(p) - p_min) > 1e-6):
            raise Fail(f"effective params mismatch (server log): n_max={n_max} n_min={n_min} p_min={p} vs k={k} p_min={p_min}", "integrity")
    if kv_type:
        # The target cache is allocated first; a later llama_kv_cache line is the draft cache, whose type is an arm knob.
        first = re.search(r"llama_kv_cache: size =[^\n]*", text)
        kinds = KV_TYPES[0].findall(text) + (KV_TYPES[1].findall(first.group()) if first else [])
        if any(kind != kv_type for kind in kinds):
            raise Fail(f"effective params mismatch: target KV type {sorted(set(kinds))}", "integrity")
    return len(seen)


def verify_log_patterns(text, patterns, what):
    """A knob that only shows in the log engaged iff each of its patterns is there."""
    missing = [p for p in patterns if not re.search(p, text)]
    if missing:
        raise Fail(f"path-engaged proof failed: {what} log lacks {missing}", "integrity")


# ---------------------------------------------------------------- sufficiency

def draft_length_check(k, p_min, drafts, draft_tokens, min_drafts):
    """With n_min=0, p_min=0 every round drafts exactly k (S5 control: 5.000 over 24 tasks); p_min>0 must shorten drafts."""
    if not drafts or drafts < min_drafts:
        return "insufficient", f"{drafts or 0} verify steps < {min_drafts}", None
    mean = draft_tokens / drafts
    if mean > k + 1e-6:
        return "fail", f"mean draft length {mean:.3f} above k={k}", mean
    if p_min == 0 and mean < 0.995 * k:
        return "fail", f"p_min=0 arm drafted {mean:.3f} < k={k}: not the configured k or an early stop is active", mean
    if p_min > 0 and mean > 0.97 * k:
        return "fail", f"p_min={p_min} did not shorten drafts (mean {mean:.3f} of k={k}): flag inert", mean
    return "ok", f"mean draft length {mean:.3f} of k={k}", mean


def steady_intervals(record, minimum=40, k=5, window=None):
    """Full-K interior rounds from cumulative per-token timings: a round of k drafts ends where draft_n crosses a multiple
    of k. Rows without the four counters are skipped (real streams carry empty and partial chunks). window = (lo, hi) in the
    chunks' own "t" stamps keeps only rounds whose both ends arrived inside it (concurrent streams: the overlap window)."""
    chunks = [c for c in record["chunks"] if isinstance(c.get("timings"), dict) and
              all(isinstance(c["timings"].get(key), (int, float)) for key in ("predicted_ms", "predicted_n", "draft_n", "draft_n_accepted"))]
    valid = [c["timings"] for c in chunks]
    if not valid:
        raise Fail("missing final timing counters", "insufficient_data")
    final = valid[-1]
    if any(v != v or v in (float("inf"), float("-inf")) or v < 0 for v in final.values() if isinstance(v, (int, float))):
        raise Fail("invalid timing", "deterministic")
    if final["draft_n"] != int(final["draft_n"]) or not 0 <= final["draft_n_accepted"] <= final["draft_n"]:
        raise Fail("invalid draft counters", "deterministic")
    points, last_n, last_ms = {}, 0, 0
    for c in chunks:
        t = c["timings"]
        n, ms = t["draft_n"], t["predicted_ms"]
        if n < last_n or ms < last_ms:
            raise Fail("reversed cumulative timing", "deterministic")
        last_n, last_ms = n, ms
        if window is not None and not window[0] <= c.get("t", -1) <= window[1]:
            continue
        if n > 0 and n % k == 0 and t["predicted_n"] <= final["predicted_n"] - (k + 1):
            points[n // k] = ms
    keys = sorted(points)
    intervals = [points[b] - points[a] for a, b in zip(keys, keys[1:]) if a >= 3 and b == a + 1 and points[b] > points[a]]
    if len(intervals) < minimum:
        raise Fail(f"insufficient steady full-K interior intervals: {len(intervals)} < {minimum}", "insufficient_data")
    return {"intervals_M": len(intervals), "steady_ms_per_round_D": statistics.mean(intervals), "timings": final}


def token_identity(record):
    """A streamed final chunk carries no tokens (server-context.cpp send_final_response): concatenate the partial chunks."""
    chunks = record["chunks"]
    final = next((c for c in reversed(chunks) if c.get("stop") is True), None)
    tokens = [t for c in chunks for t in (c.get("tokens") or [])]
    if final is None or not tokens:
        raise Fail("missing final token identity", "insufficient_data")
    return tokens, {k: final["timings"].get(k) for k in ("predicted_n", "draft_n", "draft_n_accepted", "prompt_n", "cache_n")}


# Trace writer cost measured 2026-10-07 on stack6; remeasure after the writer-thread fix.
TRACE_MS_PER_KB = 0.0043
TRACE_DELTA_KB_MAX = 5
# G: conservative reference until an A/A report measures this shape.
AA_BOOT_SD_MS = {(1, False): 0.10, (2, False): 0.10, (1, True): 0.10, (2, True): 0.10}


def boot_delta(runs_by_arm, streams=1, metric="steady_ms_per_round_D"):
    """Arms in insertion order; every later arm against the first ("pairs"), the first pair also at top level.
    streams share batch steps, so average them inside each boot.

    Pair boot i for the descriptive delta list/SD. SE uses independent arm variances; pairing by
    ordinal alone does not establish covariance (graph-capture untraced reference, 2026-10-07).
    """
    values = {a: [statistics.mean(s[metric] for s in r["streams"]) if streams > 1 else r[metric]
                  for r in runs] for a, runs in runs_by_arm.items()}
    first, *rest = values

    def pair(up, patch):
        paired = [b - a for a, b in zip(up, patch)] if len(up) == len(patch) else []
        enough = min(len(up), len(patch)) >= 2
        return {"delta": statistics.mean(patch) - statistics.mean(up), "boots": paired,
                "sd": statistics.stdev(paired) if len(paired) > 1 else None,
                "se": math.sqrt(statistics.variance(up) / len(up) + statistics.variance(patch) / len(patch)) if enough else None}

    pairs = {a: pair(values[first], values[a]) for a in rest}
    return pairs[rest[0]] | {"arms": values, "pairs": pairs}


def request_split(timings, k, steady_ms):
    """D: draft_n/k includes a possibly partial last round (at most one round of error)."""
    return {"prompt_ms_M": timings["prompt_ms"],
            "fixed_ms_D": timings["predicted_ms"] - timings["draft_n"] / k * steady_ms}


def trace_bytes_per_round(boot_dir, rounds):
    """Whole-boot bytes / whole-boot rounds; never read the trace body."""
    path = Path(boot_dir) / "round-trace.jsonl"
    return path.stat().st_size / rounds if path.exists() and rounds > 0 else None


# ---------------------------------------------------------------- engagement proofs

MASK_ENTRY = re.compile(rb"\[attn_inp_kq_mask \(\s*(\d+)([KM])\)\]")


def mask_uploads(path, start=0, end=None):
    """KiB of every attn_inp_kq_mask split input the scheduler logged in [start, end) (GGML_SCHED_DEBUG at verbosity 5)."""
    sizes, left = [], None if end is None else end - start
    with open(path, "rb") as stream:
        stream.seek(start)
        for raw in stream:
            if left is not None:
                if left <= 0:
                    break
                raw, left = raw[:left], left - len(raw)
            if b"## SPLIT #" in raw:
                sizes += [int(v) * (1024 if u == b"M" else 1) for v, u in MASK_ENTRY.findall(raw)]
    return sizes


def check_compact_masks(sizes, limit):
    if len(sizes) < 2:
        raise Fail(f"path-engaged proof inconclusive: {len(sizes)} attn_inp_kq_mask uploads in the deep decode window (debug log off?)", "insufficient_data")
    big = sorted(set(s for s in sizes if s > limit))
    if big:
        raise Fail(f"path-engaged proof failed: full mask uploads {big[:6]} KiB > {limit} KiB (compact suffix path not engaged)", "integrity")
    return {"uploads": len(sizes), "max_kib": max(sizes), "sizes_kib": sorted(set(sizes))}


def trace_counts(path):
    counts = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return counts
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for phase in row.get("phases", []):
            for counter in phase.get("counters", []):
                counts[counter["kind"]] = counts.get(counter["kind"], 0) + counter["count"]
    return counts


@functools.lru_cache(maxsize=2)
def _trace_rounds(path, size, mtime_ns):
    """(begin_ns, [(kind, count, host_us), ...]) per round, counters in trace order so window sums stay bit-identical.
    Keyed on size and mtime so a trace that grew is parsed again; a boot's trace is GBs, re-read once per depth before."""
    del size, mtime_ns
    rounds = []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            counters = [(c["kind"], c["count"], c.get("host_us", 0.0))
                        for phase in row.get("phases", []) for c in phase.get("counters", [])]
            rounds.append((row.get("phases", [{}])[0].get("begin_ns", -1), counters))
    return rounds


def trace_window(path, lo_ns, hi_ns, kinds):
    """Counters of the rounds that began inside [lo_ns, hi_ns]. The trace stamps CLOCK_MONOTONIC, the clock of
    time.monotonic_ns(); a window that catches no round is inconclusive, never a zero."""
    rounds, out = 0, {k: {"count": 0, "host_us": 0.0} for k in kinds}
    try:
        st = Path(path).stat()
        parsed = _trace_rounds(str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        parsed = []
    for begin_ns, counters in parsed:
        if not lo_ns <= begin_ns <= hi_ns:
            continue
        rounds += 1
        for kind, count, host_us in counters:
            if kind in out:
                out[kind]["count"] += count
                out[kind]["host_us"] += host_us
    if not rounds:
        raise Fail(f"counters inconclusive: no round-trace round inside the request window of {Path(path).parent.name}", "insufficient_data")
    return {"rounds": rounds, **out}


def check_staging(counts):
    staged, sync = counts.get("input_fence_staged", 0), counts.get("input_fence_sync", 0)
    if staged + sync == 0:
        raise Fail("staging proof inconclusive: the round trace holds no input_fence counters", "insufficient_data")
    if staged < 1 or sync > staged:
        raise Fail(f"staging-engaged proof failed: input_fence_staged {staged}, input_fence_sync {sync}", "integrity")
    return {"input_fence_staged": staged, "input_fence_sync": sync, "graph_hit": counts.get("graph_hit", 0)}


# ---------------------------------------------------------------- processes and HTTP

def http_json(url, body=None, timeout=30):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def call(url, body=None, timeout=30):
    """No harness wall clock: gpuq's RuntimeMaxSec is the only timeout, so a slow or failed request is transient."""
    try:
        return http_json(url, body, timeout)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise Fail(f"request failed: {type(error).__name__}: {error}", "transient")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _own_group(launcher):
    """The server dies with its launcher (PR_SET_PDEATHSIG) and gets its own process group for a clean group kill."""
    os.setsid()
    import ctypes
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != launcher:
        os._exit(126)


class Server:
    """One server process. The caller decides argv/env and records them first; this only starts, waits and stops."""

    def __init__(self, argv, env, log_path):
        self.argv, self.env, self.log = [str(a) for a in argv], env, Path(log_path)
        self.proc = None

    def start(self):
        if Path(self.argv[0]).name == SERVER_NAME:
            raise Fail(f"effective params mismatch: refusing to execute a file named {SERVER_NAME}", "integrity")
        launcher = os.getpid()
        with self.log.open("ab") as stream:
            self.proc = subprocess.Popen(self.argv, env=self.env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         preexec_fn=lambda: _own_group(launcher))
        return self.proc.pid

    def wait_ready(self, base, cap_s):
        until = time.time() + cap_s
        while time.time() < until:
            if self.proc.poll() is not None:
                raise Fail(f"server exited rc {self.proc.returncode} during boot: {self.tail()}", "transient")
            try:
                if http_json(base + "/health", timeout=2).get("status") == "ok":
                    return
            except (OSError, ValueError):
                time.sleep(0.5)
        raise Fail(f"server not ready inside {cap_s}s", "transient")

    def tail(self, n=500):
        try:
            return self.log.read_text(errors="replace")[-n:]
        except OSError:
            return ""

    def stop(self, grace=60):
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=grace)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def stop_child(child, grace):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def process_nice(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[16])
    except (OSError, IndexError, ValueError):
        return None


def libs_outside(pid, root):
    """Engine libraries the live server mapped from outside its frozen root (a stray LD_LIBRARY_PATH or RUNPATH)."""
    libs = {line.split()[-1] for line in Path(f"/proc/{pid}/maps").read_text().splitlines()
            if re.search(r"/lib(llama|ggml|mtmd)[^/]*\.so", line)}
    return sorted(p for p in libs if not Path(p).resolve().is_relative_to(Path(root).resolve()))


def start_smi(path, cmd=None, interval_ms=500):
    """Background free-VRAM sampler for one boot: `uuid, MiB free` rows, killed by stop_child. The caller reads it with min_free_vram."""
    cmd = cmd or ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits", "-lms", str(interval_ms)]
    with open(path, "wb") as stream:
        return subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def min_free_vram(path, devices):
    """{uuid: lowest MiB free} over the sampler's rows. A selected GPU with no row is missing data, never a 0 or a pass."""
    low = {}
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        uuid, _, free = (part.strip() for part in line.partition(","))
        try:
            low[uuid] = min(low.get(uuid, float("inf")), int(float(free)))
        except ValueError:
            continue  # a header, a half-written last row, an error line
    missing = [d for d in devices if d not in low]
    if missing:
        raise Fail(f"VRAM samples missing for {missing}", "insufficient_data")
    return {d: low[d] for d in devices}


def _meminfo_kib():
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        out[key] = int(rest.split()[0])
    return out


def _proc_kib(pid, key):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return None


def _read_meminfo():
    m = _meminfo_kib()
    return {"avail_mib": m["MemAvailable"] // 1024, "swap_used_mib": (m["SwapTotal"] - m["SwapFree"]) // 1024}


def _read_zram():
    stat = Path("/sys/block/zram0/mm_stat").read_text().split()
    return {"zram_orig_mib": int(stat[0]) // 2**20, "zram_used_mib": int(stat[2]) // 2**20}


def _read_vmstat():
    pairs = (line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {k: int(v) for k, v in pairs if k in ("pswpin", "pswpout")}


def mem_sample(pid=None):
    """One host-memory row: MemAvailable, swap in use, zram (original/compressed), cumulative swap-in/out pages, and the
    server's VmRSS/VmSwap. A source that cannot be read is absent from the row, never 0."""
    row = {"t": round(time.time(), 1)}
    readers = [_read_meminfo, _read_zram, _read_vmstat]
    if pid:
        readers.append(lambda: {"rss_mib": _proc_kib(pid, "VmRSS") // 1024, "vmswap_mib": _proc_kib(pid, "VmSwap") // 1024})
    for read in readers:
        try:
            row |= read()
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
    return row


class MemLog:
    """Host-memory sampler for one boot (attribution only, never an exit code): the host prompt cache and co-tenants swap the
    host, and swap contaminates timing. Rows go to `path` (JSONL); stop() returns mem_summary(path). The pid is read each tick
    from `pid_fn`, so a server that starts after the sampler is picked up."""

    def __init__(self, path, pid_fn=lambda: None, interval_s=2.0):
        self.path, self.pid_fn, self.interval, self._stop = Path(path), pid_fn, interval_s, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self.path.open("w") as stream:
            while True:
                stream.write(json.dumps(mem_sample(self.pid_fn())) + "\n")
                stream.flush()
                if self._stop.wait(self.interval):
                    return

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        return mem_summary(self.path)


def mem_summary(path):
    """First/last/extreme of a MemLog file. swap_out_mib/swap_in_mib: pages moved during the boot (x4 KiB); peak_*: the high-water marks."""
    rows = []
    try:
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # half-written last row
    except OSError:
        pass
    col = lambda key: [r[key] for r in rows if key in r]
    out = {"samples": len(rows)}
    for key, name, pick in (("swap_used_mib", "peak_swap_used_mib", max), ("zram_used_mib", "peak_zram_used_mib", max), ("rss_mib", "peak_rss_mib", max),
                            ("vmswap_mib", "peak_vmswap_mib", max), ("avail_mib", "min_avail_mib", min)):
        if col(key):
            out[name] = pick(col(key))
    if col("swap_used_mib"):
        out["swap_used_start_mib"], out["swap_used_end_mib"] = col("swap_used_mib")[0], col("swap_used_mib")[-1]
    for key, name in (("pswpout", "swap_out_mib"), ("pswpin", "swap_in_mib")):
        if len(col(key)) > 1:
            out[name] = (col(key)[-1] - col(key)[0]) * 4 // 1024
    return out


def gpu_memory(raw, devices):
    """MiB used on the selected GPUs from `nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv`."""
    found = {}
    for row in csv.DictReader(io.StringIO(raw)):
        row = {k.strip(): v.strip() for k, v in row.items()}
        found[row["uuid"]] = int(row["memory.used [MiB]"].split()[0])
    if not set(devices) <= found.keys():
        raise Fail("selected GPU missing", "transient")
    return sum(found[d] for d in devices)


def wait_gpu_idle(devices, limit_mib=1500, wait_s=90):
    """Between arms only: the previous server frees VRAM asynchronously and gpuq does not gate there."""
    end = time.time() + wait_s
    while True:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv"], capture_output=True, text=True, timeout=30)
        used = gpu_memory(p.stdout, devices) if p.returncode == 0 else None
        if used is not None and used <= limit_mib:
            return used
        if time.time() > end:
            raise Fail(f"GPUs still hold {used} MiB after the previous server stopped", "transient")
        time.sleep(3)


def record_journal(since, path, journal_file=None):
    """Attribution only: keeps the kernel journal window of one arm and counts GPU-fault lines. Never decides an exit code:
    gpuq's post-run check owns the hazard verdict. Returns the count, or None when the journal could not be read."""
    try:
        if journal_file:
            text = Path(journal_file).read_text()
        else:
            p = subprocess.run(["journalctl", "-k", "--since", f"@{since:.0f}", "--no-pager"], capture_output=True, text=True, timeout=30)
            if p.returncode:
                raise OSError(f"journalctl rc {p.returncode}: {p.stderr[-200:]}")
            text = p.stdout  # "-- No entries --" is a quiet window, not an unreadable one
        Path(path).write_text(text)
        return len(re.findall(XID_RE, text))
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as error:
        try:
            Path(path).write_text(f"unreadable: {error}\n")
        except OSError:
            pass
        return None


def run_main(fn, out=None):
    """The one place a Fail becomes an exit code; anything else is a harness bug (deterministic), never a silent 1."""
    try:
        return fn()
    except Fail as error:
        log(out, f"FAILED {error.cls} rc={error.rc}: {error}")
        return error.rc
    except KeyboardInterrupt:
        log(out, "signal received; stopped own processes")
        return 130
    except Exception as error:
        import traceback
        traceback.print_exc()
        log(out, f"FAILED deterministic rc={RC['deterministic']}: harness error {type(error).__name__}: {error}")
        return RC["deterministic"]
