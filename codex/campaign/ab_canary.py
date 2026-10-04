"""Canary for the stack2-metafix (A) vs stack3-mask (B) A/B: fresh boots, cross-arm identity on a seeded deep high-entropy
request, and on B the proof that the compact mask path and input staging are really engaged. Pure functions plus a small
Boot driver, so the CPU stub can exercise every gate."""
import json
import os
import random
import re
import hashlib
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from ab_common import ARMS, CTX, EXTRA, KNOBS, MODEL, Fail, verify_launch, verify_nice

DEVICES = ["GPU-ee9a3956-297c-604b-b4c1-f07930d27e46", "GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f"]
SEED = 4242
N_PREDICT = 160
DEEP_N_PREDICT = 512  # crosses several 256-cell n_kv pads, which forces fresh decode-shape graph builds under the debug log
TARGET_PROMPT_TOKENS = int(os.environ.get("AB_PROMPT_TOKENS", "33800"))  # just over the 32768 compact-path threshold: prefill dominates the canary wall
MIN_DEEP_KV = 32768  # the compact path only engages from here
MAX_SUFFIX_KIB = 8  # one verify batch of 6 rows is 3 KiB, a draft row 0.5 KiB; a full mask at this depth is >= 64 KiB
DEBUG_FLAGS = (("--verbosity", "5"),)
DEBUG_ENV = {"GGML_SCHED_DEBUG": "1"}
BOOT_CAP_S = int(os.environ.get("AB_BOOT_CAP_S", "420"))
REQUEST_CAP_S = int(os.environ.get("AB_REQUEST_CAP_S", "900"))
SYL = ["ka", "to", "mi", "ra", "zu", "pe", "lo", "ve", "ni", "sa", "du", "fo", "gi", "bu", "he", "wo", "xa", "ye", "qu", "ja"]


def http(url, body=None, timeout=30):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def call(url, body=None, timeout=30, deadline=None):
    """http() with failures mapped to exit codes: past the canary wall cap -> 7, any transport failure before it -> 4 (transient)."""
    left = None if deadline is None else deadline - time.time()
    if left is not None and left <= 0:
        raise Fail("canary hit its wall cap", 7)
    try:
        return http(url, body, timeout if left is None else min(timeout, left))
    except (OSError, ValueError, socket.timeout, urllib.error.URLError) as error:
        if deadline is not None and time.time() >= deadline - 1:
            raise Fail("canary hit its wall cap", 7)
        raise Fail(f"canary request failed: {type(error).__name__}: {error}", 4)


def words(count, rng):
    return " ".join("".join(rng.choice(SYL) for _ in range(rng.choice((2, 3, 3, 4)))) for _ in range(count))


def build_prompt(tokenize, target=TARGET_PROMPT_TOKENS):
    """Pseudo-random invented words (never a copy or repeated-token prompt), deterministic given the tokenizer; returns the token ids."""
    rng = random.Random(SEED)
    text = words(4000, rng)
    toks = tokenize(text)
    for _ in range(12):
        if len(toks) >= target:
            break
        per_word = len(toks) / max(text.count(" ") + 1, 1)
        text += " " + words(max(200, int((target - len(toks)) / max(per_word, 0.1))), rng)
        toks = tokenize(text)
    if not target <= len(toks) <= target * 1.25:
        raise Fail(f"canary prompt has {len(toks)} tokens, wanted {target}..{int(target * 1.25)}", 5)
    return toks


def server_argv(arm, model, port, debug=False):
    spec = ARMS[arm]
    cmd = [str(spec["runner_binary"]), "-m", str(model), "--alias", "local-benchmark", "-c", str(CTX), "-np", "1", "-ctk", "q8_0", "-ctv", "q8_0",
           "-ngl", "99", "-fa", "on", "--spec-type", "draft-mtp", "--spec-draft-n-max", str(spec["k"]), "--spec-draft-n-min", "0",
           "--spec-draft-p-min", str(spec["p_min"]), "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16", "-sm", "tensor",
           "--tensor-split", "50,50", "-b", "2048", "-ub", "128", "--host", "127.0.0.1", "--port", str(port), "--metrics",
           "--temp", "1", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.05", *EXTRA]
    for flag, value in (DEBUG_FLAGS if debug else ()):
        cmd += [flag, value]
    return cmd


class Boot:
    """One fresh server for one arm. The environment is exactly the arm's (+ the debug knobs on the proof boot); nothing changes the nice."""

    def __init__(self, ctx, arm, out, debug=False, stub=False, model=MODEL, deadline=None):
        self.deadline = deadline
        self.ctx, self.arm, self.out, self.debug, self.stub, self.model = ctx, arm, Path(out), debug, stub, model
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = self.out / "server.log"
        self.proc = None
        self.requests = 0

    def env(self):
        spec = ARMS[self.arm]
        base = {k: v for k, v in os.environ.items() if k not in KNOBS and k != "LD_LIBRARY_PATH"}
        env = base | spec["env"]
        if self.debug:
            env |= DEBUG_ENV | {"LLAMA_ROUND_TRACE": str(self.out / "rounds.jsonl")}
        if self.stub:
            env |= {k: v for k, v in os.environ.items() if k.startswith("STUB_")} | {"STUB_ARM": self.arm}
            if self.arm == "B" and env.get("STUB_B_NO_STAGING") == "1":
                env.pop("GGML_SCHED_INPUT_STAGING", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(DEVICES)
        return env

    def expected_env(self):
        return ARMS[self.arm]["env"] | (DEBUG_ENV | {"LLAMA_ROUND_TRACE": str(self.out / "rounds.jsonl")} if self.debug else {})

    def start(self):
        spec = ARMS[self.arm]
        command = server_argv(self.arm, self.model, self.ctx.port, self.debug)
        executed = [str(Path(__file__).with_name("stub_srv.py")), *command[1:]] if self.stub else [str(spec["exe"]), *command[1:]]
        env = self.env()
        record = {k: env.get(k) for k in (*KNOBS, "LD_LIBRARY_PATH")}
        nice = os.getpriority(os.PRIO_PROCESS, 0)
        verify_launch(command, self.arm, None if self.stub else executed, extra_ok=DEBUG_FLAGS if self.debug else (), agentic=True)
        verify_env_expected(record, self.expected_env())
        if not self.stub or os.environ.get("AB_STUB_CHECK_NICE") == "1":
            verify_nice(nice)
        (self.out / "launch.json").write_text(json.dumps({"command": command, "executed_command": executed, "executed_env": record,
                                                          "server_nice": nice, "debug": self.debug}, indent=1))
        parent = os.getpid()

        def child_setup():
            os.setsid()
            import ctypes
            if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent:
                os._exit(126)

        with self.log.open("ab") as stream:
            self.proc = subprocess.Popen(executed, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, preexec_fn=child_setup)
        base = f"http://127.0.0.1:{self.ctx.port}"
        deadline = time.time() + BOOT_CAP_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise Fail(f"canary server for arm {self.arm} exited rc {self.proc.returncode} during boot: {self.tail()}", 4)
            try:
                if http(base + "/health", timeout=2).get("status") == "ok":
                    return base
            except (OSError, ValueError):
                time.sleep(1.0)
            if self.deadline and time.time() >= self.deadline:
                raise Fail("canary hit its wall cap", 7)
        raise Fail(f"canary server for arm {self.arm} not ready inside {BOOT_CAP_S}s", 4)

    def tail(self, n=500):
        try:
            return self.log.read_text(errors="replace")[-n:]
        except OSError:
            return ""

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def server_nice_matches(pid, expected):
    """The live server's nice equals the orchestrator's: proof the child did not inherit a lower priority."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[16]) == expected
    except (OSError, IndexError, ValueError):
        return False


def verify_env_expected(record, want):
    problems = [f"{k}={record.get(k)!r} != {want.get(k)!r}" for k in (*KNOBS, "LD_LIBRARY_PATH") if record.get(k) != want.get(k)]
    if problems:
        raise Fail("effective params mismatch (env): " + "; ".join(problems), 6)


def complete(base, prompt, n_predict, deadline=None, seed=SEED):
    body = {"prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "cache_prompt": True, "seed": seed, "temperature": 1.0,
            "top_k": 20, "top_p": 0.95, "min_p": 0.05, "stream": False, "return_tokens": True}
    reply = call(base + "/completion", body, REQUEST_CAP_S, deadline)
    tokens = reply.get("tokens")
    timings = reply.get("timings") or {}
    if not tokens or any(not isinstance(timings.get(k), (int, float)) for k in ("predicted_n", "draft_n", "draft_n_accepted", "prompt_n", "cache_n")):
        raise Fail("missing final token identity or draft counters in the completion reply", 5)
    if len(tokens) != timings["predicted_n"] or len(tokens) != n_predict:
        raise Fail(f"completion returned {len(tokens)} tokens (predicted_n {timings['predicted_n']}), wanted {n_predict}", 5)
    return {"tokens": tokens, "timings": {k: timings.get(k) for k in ("predicted_n", "draft_n", "draft_n_accepted", "prompt_n", "cache_n")}}


# ---------------------------------------------------------------- gates

SPLIT_LINE = re.compile(r"## SPLIT #\d+: [^\n]*")
MASK_ENTRY = re.compile(r"\[attn_inp_kq_mask \(\s*(\d+)([KM])\)\]")


def mask_uploads(path, offset=0):
    """KiB of every attn_inp_kq_mask split input the scheduler logged after `offset` (GGML_SCHED_DEBUG, --verbosity 5); streamed, the log is large."""
    sizes = []
    with open(path, "rb") as stream:
        stream.seek(offset)
        for raw in stream:
            if b"## SPLIT #" not in raw or b"attn_inp_kq_mask" not in raw:
                continue
            for value, unit in MASK_ENTRY.findall(raw.decode(errors="replace")):
                sizes.append(int(value) * (1024 if unit == "M" else 1))
    return sizes


def check_compact_masks(sizes, limit=MAX_SUFFIX_KIB):
    if len(sizes) < 2:
        raise Fail(f"path-engaged proof inconclusive: {len(sizes)} attn_inp_kq_mask uploads in the deep decode log (debug log off or no fresh graph)", 5)
    big = sorted(set(s for s in sizes if s > limit))
    if big:
        raise Fail(f"path-engaged proof failed: deep decode uploads full masks, sizes KiB {big[:6]} > {limit} KiB (compact suffix path not engaged)", 6)
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


def check_staging(counts):
    staged, sync = counts.get("input_fence_staged", 0), counts.get("input_fence_sync", 0)
    if staged + sync == 0:
        raise Fail("staging proof inconclusive: the round trace holds no input_fence counters", 5)
    if staged < 1 or sync > staged:
        raise Fail(f"staging-engaged proof failed: input_fence_staged {staged}, input_fence_sync {sync}", 6)
    return {"input_fence_staged": staged, "input_fence_sync": sync, "graph_hit": counts.get("graph_hit", 0)}


def check_identity(a, b):
    if a["tokens"] != b["tokens"]:
        first = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), min(len(a["tokens"]), len(b["tokens"])))
        raise Fail(f"cross-arm identity failed: tokens differ at {first} of {len(a['tokens'])}/{len(b['tokens'])}", 6)
    for key in ("predicted_n", "draft_n", "draft_n_accepted"):
        if a["timings"][key] != b["timings"][key] or a["timings"][key] is None:
            raise Fail(f"cross-arm identity failed: {key} {a['timings'][key]} != {b['timings'][key]}", 6)
    return {"tokens": len(a["tokens"]), "draft_n": a["timings"]["draft_n"], "draft_n_accepted": a["timings"]["draft_n_accepted"]}


def check_deep(reply):
    t = reply["timings"]
    depth = (t.get("prompt_n") or 0) + (t.get("cache_n") or 0)
    if depth < MIN_DEEP_KV:
        raise Fail(f"canary request reached only {depth} cached+prompt tokens < {MIN_DEEP_KV}", 5)
    return depth


# ---------------------------------------------------------------- flow

def wait_gpu_idle(stub, limit_mib=1500, wait_s=90):
    """The previous server's VRAM is released asynchronously; the next fresh boot must not start on top of it."""
    if stub:
        return
    end = time.time() + wait_s
    while True:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        used = sum(int(v) for v in p.stdout.split()) if p.returncode == 0 else None
        if used is not None and used <= limit_mib:
            return
        if time.time() > end:
            raise Fail(f"GPUs still hold {used} MiB after the previous canary server stopped", 4)
        time.sleep(3)


def one_arm(ctx, arm, out, stub, deadline, log):
    """Fresh boot -> deep seeded request (identity) -> same prompt again from the cache (decode proofs on B). Returns the identity record."""
    boot = Boot(ctx, arm, out / f"boot-{arm}", debug=arm == "B", stub=stub, deadline=deadline)
    try:
        base = boot.start()
        log(f"canary arm {arm}: fresh boot ready")
        toks = build_prompt(lambda text: call(base + "/tokenize", {"content": text}, 300, deadline)["tokens"])
        digest = hashlib.sha256(json.dumps(toks).encode()).hexdigest()
        r1 = complete(base, toks, N_PREDICT, deadline)
        record = {"arm": arm, "prompt_tokens": len(toks), "prompt_sha": digest, "r1": r1, "depth": check_deep({"timings": {"prompt_n": r1["timings"]["prompt_n"], "cache_n": r1["timings"]["cache_n"]}})}
        log(f"canary arm {arm}: R1 {r1['timings']} prompt sha {digest[:12]}")
        mark = boot.log.stat().st_size
        r2 = complete(base, toks, DEEP_N_PREDICT, deadline)
        record["r2"] = {"timings": r2["timings"], "tokens": len(r2["tokens"])}
        # n_kv of every R2 decode step is >= the cached prefix, so the compact-mask path must be live for B
        if r2["timings"]["cache_n"] < MIN_DEEP_KV:
            raise Fail(f"R2 reused only {r2['timings']['cache_n']} cached tokens < {MIN_DEEP_KV}: no deep decode to inspect", 5)
        if arm == "B":
            record["mask"] = check_compact_masks(mask_uploads(boot.log, mark))
            record["staging"] = check_staging(trace_counts(boot.out / "rounds.jsonl"))
            log(f"canary arm B: compact mask uploads {record['mask']['uploads']} (max {record['mask']['max_kib']} KiB, sizes {record['mask']['sizes_kib']}); staging {record['staging']}")
        if boot.proc.poll() is not None:
            raise Fail(f"canary server for arm {arm} died: {boot.tail()}", 4)
        if not stub and not server_nice_matches(boot.proc.pid, os.getpriority(os.PRIO_PROCESS, 0)):
            raise Fail("effective params mismatch (nice): the live server's nice differs from the orchestrator's", 6)
        return record
    finally:
        boot.stop()


def proofs(ctx, out, stub, deadline, log):
    out = Path(out)
    records = {}
    for arm in ("A", "B"):
        wait_gpu_idle(stub)
        records[arm] = one_arm(ctx, arm, out, stub, deadline, log)
    a, b = records["A"], records["B"]
    if a["prompt_sha"] != b["prompt_sha"]:
        raise Fail("canary prompt differs between arms (tokenizer mismatch)", 6)
    identity = check_identity(a["r1"], b["r1"])
    log(f"canary identity: {identity['tokens']} tokens byte-identical, draft_n {identity['draft_n']} accepted {identity['draft_n_accepted']}")
    wait_gpu_idle(stub)
    summary = {"identity": identity, "A": {k: v for k, v in a.items() if k != "r1"}, "B": {k: v for k, v in b.items() if k != "r1"}}
    (out / "canary-proofs.json").write_text(json.dumps(summary, indent=1))
    return summary
