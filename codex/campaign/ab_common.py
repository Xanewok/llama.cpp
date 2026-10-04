"""Shared contract for the stack2-metafix (A) vs stack3-mask (B) campaign A/B items: arms with their own frozen roots, expected server argv, metric derivation, effectiveness checks."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = Path.home() / "repos/qwen38-eval"
WEEKEND = Path.home() / "repos/qwen38-weekend"
STACK3_MASK = Path.home() / "repos/stack3-mask"
RUNNER = EVAL / "scripts/experiments/dual_5060ti/run_swift_longtail.py"
ASSETS = EVAL / "results/2026-09-12-mixed-pair/campaign-confirmed-fullsuite-20260914-r8/snapshot/agentic-assets"
IDS = Path.home() / "models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json"
IDS_SHA = "e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6"
MEMOC_GUARD = Path.home() / "repos/gpuq/vendor/memoc/memoc_guard.sh"
XID_SINCE = "2026-10-03 18:57:00"
XID_RE = r"\bXid\b|fallen off the bus"

# The model is the generator parameter (AB_ITEM): one item per model, same arms, same flags.
MODELS = {
    "iq4": {"id": "swift15-iq4-xs", "path": Path.home() / "models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf", "bytes": 15475951296,
            "sha": "e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df", "label": "IQ4_XS"},
    "j5q": {"id": "swift15-weekend-f2c850ea3cd8", "path": Path("/mnt/scratch/work/models/j5/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf"),
            "bytes": 19996051744, "sha": "f2c850ea3cd818b9d669fbd1e18767ef669ce012e614d819cce5ab6809e3a774", "label": "J5q"},
}
ITEM_KEY = os.environ.get("AB_ITEM", "iq4")
MODEL_SPEC = MODELS[ITEM_KEY]
MODEL = MODEL_SPEC["path"]
MODEL_BYTES = MODEL_SPEC["bytes"]
MODEL_SHA = MODEL_SPEC["sha"]
MODEL_ID = MODEL_SPEC["id"]

# Env knobs that change the serving path; each arm must set exactly its own and nothing else.
KNOBS = ("LLAMA_MTP_GRAPH_REUSE", "GGML_META_PLAN_REUSE", "GGML_SCHED_INPUT_STAGING", "LLAMA_GRAPH_REUSE_DISABLE", "LLAMA_ROUND_TRACE",
         "GGML_SCHED_DEBUG", "GGML_CUDA_FA_MMA_Q8_0")

_W = WEEKEND / "frozen/stack2-metafix"
_B = STACK3_MASK / "codex/frozen/stack3-mask"
_BR = STACK3_MASK / "codex/frozen/stack3-mask-runroot"
# root: the frozen tree whose SHA256SUMS pins the bytes; runner_root: the tree the shared runner treats as the engine checkout
# (build/bin/llama-server, gguf-py, commit); exe: the file actually executed (never named llama-server).
ARM_SPECS = {
    "A": {"label": "stack2-metafix", "k": 5, "p_min": 0, "commit": "c23324d5f9f0fc00ad3f2e6441e39ba061f41544",
          "root": _W, "bin": _W / "build/bin", "runner_root": _W, "exe": _W / "build/bin/weekend-srv", "env": {},
          "runner_binary": _W / "build/bin/llama-server", "sums": [_W / "build/bin/SHA256SUMS"]},
    "B": {"label": "stack3-mask+staging", "k": 5, "p_min": 0, "commit": "45878736633c181bf040546c13a8052181730f67",
          "root": _B, "bin": _B / "bin", "runner_root": _BR, "exe": _B / "bin/round-trace",
          "env": {"GGML_SCHED_INPUT_STAGING": "1", "LD_LIBRARY_PATH": str(_B / "bin")},
          "runner_binary": _BR / "build/bin/llama-server", "sums": [_B / "SHA256SUMS", _B / "bin/SHA256SUMS", _BR / "RUNROOT.SHA256SUMS"]},
}
# Pinned sums reported when the arm roots were frozen (checked by cpu_checks on top of sha256sum -c).
PINNED_ROOT_SUMS = {
    str(_B / "SHA256SUMS"): "cb2327c2a42e52d3afdae18e7da5a7f175e42a16056fbf12757db7208a3fce08 SOURCE\n"
                            "e98a3e498fa551959a417907e4d76b34f21c69a2eb6eb57a57b7b982565077d5 bin/SHA256SUMS\n",
}
IDLE_NICE = 0

CTX = 102400
MAX_K = 7  # k >= 8 makes the verify batch exceed MMVQ_MAX_BATCH_SIZE (8 rows) and switches kernels
ARMS = ARM_SPECS
TASKS = ["rust/doubly-linked-list", "rust/book-store", "rust/react", "rust/alphametics", "rust/forth",
         "rust/xorcism", "rust/nucleotide-codons", "javascript/complex-numbers"]
FIT_TASK = "javascript/affine-cipher"
CANARY_TASK = "javascript/complex-numbers"
RUNAWAY_TOKENS = 32000
MIN_DRAFTS = 200
# Failures that are model outcomes (e.g. a repetition runaway overflowing the context), kept as data.
MODEL_OUTCOME_REASONS = {"probe_failed", "probe_incomplete"}

EXTRA = ["--spec-draft-sampling", "probabilistic", "--spec-draft-backend-sampling", "--backend-sampling",
         "-t", "8", "-tb", "8", "--spec-draft-vocab", str(IDS)]


class Fail(Exception):
    """rc is the item exit code the failure maps to (see knob-real.toml exit_codes)."""

    def __init__(self, message, rc=8):
        super().__init__(message)
        self.rc = rc


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def flag_values(cmd, flag):
    return [cmd[i + 1] for i, token in enumerate(cmd[:-1]) if token == flag]


def strip_pairs(cmd, extra_ok):
    """Remove allowed "flag value" pairs (canary-only debug flags) so the contract check stays exact."""
    cmd, i = list(cmd), 0
    while i < len(cmd) - 1:
        if (cmd[i], cmd[i + 1]) in extra_ok:
            del cmd[i:i + 2]
        else:
            i += 1
    return cmd


def verify_launch(cmd, arm, executed=None, exec_names=None, ctx=CTX, agentic=True, extra_ok=()):
    """Raise Fail(rc=6) unless the argv the server was started with is exactly the arm's contract."""
    spec = ARMS[arm]
    cmd = strip_pairs(cmd, extra_ok)
    problems = []
    if str(cmd[0]) != str(spec["runner_binary"]):
        problems.append(f"campaign argv[0] {cmd[0]} != {spec['runner_binary']}")
    singles = {"-c": str(ctx), "-np": "1", "-ctk": "q8_0", "-ctv": "q8_0", "-ngl": "99", "-fa": "on",
               "--spec-type": "draft-mtp", "--spec-draft-n-max": str(spec["k"]), "--spec-draft-n-min": "0",
               "--spec-draft-type-k": "f16", "--spec-draft-type-v": "f16", "-sm": "tensor", "--tensor-split": "50,50",
               "-b": "2048", "-ub": "128", "--spec-draft-sampling": "probabilistic", "-t": "8", "-tb": "8",
               "--spec-draft-vocab": str(IDS)}
    for flag, want in singles.items():
        got = flag_values(cmd, flag)
        if got != [want]:
            problems.append(f"{flag} {got} != [{want}]")
    if agentic:
        for flag, want in {"--temp": "1", "--top-k": "20", "--top-p": "0.95", "--min-p": "0.05"}.items():
            if flag_values(cmd, flag) != [want]:
                problems.append(f"{flag} {flag_values(cmd, flag)} != [{want}]")
    got = flag_values(cmd, "--spec-draft-p-min")
    if len(got) != 1 or abs(float(got[0]) - spec["p_min"]) > 1e-9:
        problems.append(f"--spec-draft-p-min {got} != [{spec['p_min']}]")
    for flag in ("--spec-draft-backend-sampling", "--backend-sampling", "--metrics"):
        if cmd.count(flag) != 1:
            problems.append(f"{flag} count {cmd.count(flag)}")
    for flag in ("--no-spec-draft-backend-sampling", "--cache-type-k", "--cache-type-v", "--spec-draft-n-min-fallback"):
        if flag in cmd:
            problems.append(f"unexpected {flag}")
    if spec["k"] > MAX_K:
        problems.append(f"k {spec['k']} > {MAX_K}")
    if executed is not None:
        if exec_names is None and str(executed[0]) != str(spec["exe"]):
            problems.append(f"executed binary {executed[0]} != {spec['exe']}")
        elif exec_names is not None and Path(executed[0]).name not in exec_names:
            problems.append(f"executed binary {executed[0]}")
        if Path(executed[0]).name == "llama-server":
            problems.append("server named llama-server")
    if problems:
        raise Fail("effective params mismatch (argv): " + "; ".join(problems), 6)


def verify_env(env, arm):
    """The server environment is exactly the arm's: B sets input staging and its own library dir, A sets no serving knob."""
    env = env or {}
    want = ARMS[arm]["env"]
    problems = []
    for knob in KNOBS:
        if env.get(knob) != want.get(knob):
            problems.append(f"{knob}={env.get(knob)!r} != {want.get(knob)!r}")
    if env.get("LD_LIBRARY_PATH") != want.get("LD_LIBRARY_PATH"):
        problems.append(f"LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH')!r} != {want.get('LD_LIBRARY_PATH')!r}")
    if problems:
        raise Fail("effective params mismatch (env): " + "; ".join(problems), 6)


def verify_nice(nice):
    """The server must not inherit nice: the item runs un-niced and the launch record proves it."""
    if nice is None or nice > IDLE_NICE:
        raise Fail(f"effective params mismatch (nice): server nice {nice} > {IDLE_NICE}", 6)


LOG_PARAMS = re.compile(r"n_max=(\d+),? n_min=(\d+),? p_min=([\d.]+)")


def verify_server_log(text, arm):
    """Servers that echo their speculative params must agree with the arm; the real build is silent at this verbosity."""
    spec = ARMS[arm]
    seen = LOG_PARAMS.findall(text)
    for n_max, n_min, p_min in seen:
        if int(n_max) != spec["k"] or int(n_min) != 0 or abs(float(p_min) - spec["p_min"]) > 1e-6:
            raise Fail(f"effective params mismatch (server log): n_max={n_max} n_min={n_min} p_min={p_min} "
                       f"vs k={spec['k']} p_min={spec['p_min']}", 6)
    return len(seen)


def verify_props(props, ctx=CTX):
    if props.get("total_slots") != 1 or props.get("default_generation_settings", {}).get("n_ctx") != ctx:
        raise Fail(f"effective params mismatch (props): slots={props.get('total_slots')} "
                   f"n_ctx={props.get('default_generation_settings', {}).get('n_ctx')}", 6)


def draft_length_check(arm, drafts, draft_tokens):
    """Behavioural proof of the p_min/k knobs from the server's own counters: with n_min=0 and p_min=0 every round
    drafts exactly k (S5 control: 5.000 over all 24 task samples); p_min>0 must shorten the mean draft."""
    spec = ARMS[arm]
    if not drafts or drafts < MIN_DRAFTS:
        return "insufficient", f"{drafts or 0} verify steps < {MIN_DRAFTS}", None
    mean = draft_tokens / drafts
    k, p = spec["k"], spec["p_min"]
    if mean > k + 1e-6:
        return "fail", f"mean draft length {mean:.3f} above k={k}", mean
    if p == 0 and mean < 0.995 * k:
        return "fail", f"p_min=0 arm drafted {mean:.3f} < k={k}: not the configured k or an early stop is active", mean
    if p > 0 and mean > 0.97 * k:
        return "fail", f"p_min={p} did not shorten drafts (mean {mean:.3f} of k={k}): flag inert", mean
    return "ok", f"mean draft length {mean:.3f} of k={k}", mean


def derive(m):
    """Per-sample round accounting from a metrics delta. Rounds with an empty draft (p_min stopped at token 0) are
    plain decode steps the server does not count as verify steps; they emit exactly one token, so they are recovered
    as tokens - accepted - verify steps (clipped; +-~10 tokens of boundary noise per task)."""
    tokens, secs = m.get("decode_tokens"), m.get("decode_seconds")
    drafts, dtoks, acc = m.get("drafts"), m.get("draft_tokens"), m.get("draft_accepted_tokens")
    out = {"tokens": tokens, "decode_s": secs, "drafts": drafts, "draft_tokens": dtoks, "accepted": acc}
    if None in (tokens, secs, drafts, dtoks, acc):
        return out | {"plain": None, "rounds": None}
    plain = max(0.0, tokens - acc - drafts)
    return out | {"plain": plain, "rounds": drafts + plain}


def ratio(a, b):
    return a / b if a is not None and b else None


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def xid_clean(since=None, journal_file=None):
    """True when the kernel journal since the cutoff has no GPU-fault line. Unreadable parks (rc 8): rc 3 halts the queue and is reserved for a seen Xid."""
    since = since or os.environ.get("XID_SINCE") or XID_SINCE
    if journal_file:
        text = Path(journal_file).read_text()
    else:
        try:
            p = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            raise Fail(f"kernel journal unreadable: {error}", 8)
        if p.returncode:
            raise Fail("kernel journal unreadable", 8)
        text = p.stdout
        if not text.strip() or "No entries" in text:
            raise Fail("kernel journal empty since the cutoff", 8)
    return re.search(XID_RE, text, re.I) is None
