#!/usr/bin/env python3
"""Durable JSONL result emission and shared Codex invocation helpers."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tomllib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


# Bump when a field changes meaning, not when one is appended: readers tolerate a
# missing key, but not a key whose sense silently moved under them. Rows predating
# this carry null and are session-1 shapes (42, 55 and 59 fields).
SCHEMA_VERSION = 2

RESULT_FIELDS = (
    "schema_version",
    "timestamp",
    "experiment_id",
    "task_id",
    "run_index",
    "lane",
    "quant",
    "gpu",
    "gpu_count",
    "gpu_devices",
    "power_limit_w",
    "gpu_driver",
    "cuda_toolkit",
    "engine",
    "engine_commit",
    "model",
    "model_sha256",
    "served_model_path",
    "context_limit",
    "context_actual",
    "kv_k",
    "kv_v",
    "reasoning_effort",
    "temperature",
    "top_p",
    "top_k",
    "mtp_enabled",
    "mtp_nmax",
    "dflash_enabled",
    "draft_model",
    "prompt_tokens",
    "completion_tokens",
    "cached_prompt_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "prefill_tokens",
    "prefill_seconds",
    "decode_tokens",
    "decode_seconds",
    "peak_context_tokens",
    "server_high_water_tokens",
    "context_padding_tokens",
    "draft_tokens",
    "draft_accepted_tokens",
    "drafts",
    "prefill_tps",
    "decode_tps",
    "ttft_ms",
    "wall_seconds",
    "transcript",
    "max_vram_mb",
    "avg_power_w",
    "task_success",
    "tests_passed",
    "timed_out",
    "agent_turns",
    "tool_calls",
    "failed_test_cycles",
    "harness_verified",
    "human_rescue",
    "notes",
)

_CONTEXT_FIELDS = {
    "experiment_id",
    "lane",
    "quant",
    "gpu",
    "gpu_count",
    "gpu_devices",
    "power_limit_w",
    "gpu_driver",
    "cuda_toolkit",
    "engine",
    "engine_commit",
    "model",
    "model_sha256",
    "served_model_path",
    "context_limit",
    "context_actual",
    "kv_k",
    "kv_v",
    "reasoning_effort",
    "temperature",
    "top_p",
    "top_k",
    "mtp_enabled",
    "mtp_nmax",
    "dflash_enabled",
    "draft_model",
}


@dataclasses.dataclass(frozen=True, slots=True)
class RunContext:
    """Values that remain invariant for every task emitted by one process."""

    out: Path | str
    lane: str
    quant: str
    experiment_id: str | None = None
    gpu: str | None = None
    gpu_count: int | None = None
    gpu_devices: list[dict[str, object]] | None = None
    power_limit_w: float | None = None
    gpu_driver: str | None = None
    cuda_toolkit: str | None = None
    engine: str | None = "llama.cpp"
    engine_commit: str | None = None
    model: str | None = "qwen3.8-27b"
    model_sha256: str | None = None
    served_model_path: str | None = None
    context_limit: int | None = None
    context_actual: int | None = None
    kv_k: str | None = None
    kv_v: str | None = None
    reasoning_effort: str | None = "xhigh"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    mtp_enabled: bool | None = None
    mtp_nmax: int | None = None
    dflash_enabled: bool | None = None
    draft_model: str | None = None


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def build_row(
    ctx: RunContext, task_id: str, run_index: int, **measures: Any
) -> dict[str, Any]:
    """Build one fixed-shape result row without writing it."""

    if task_id is None or task_id == "":
        raise ValueError("task_id must be non-empty")
    if run_index is None:
        raise ValueError("run_index must not be null")

    unknown = set(measures) - (set(RESULT_FIELDS) - _CONTEXT_FIELDS)
    if unknown:
        raise TypeError(f"unknown result field(s): {', '.join(sorted(unknown))}")
    forbidden = set(measures) & {"schema_version", "timestamp", "task_id", "run_index"}
    if forbidden:
        raise TypeError(f"reserved result field(s): {', '.join(sorted(forbidden))}")

    row = dict.fromkeys(RESULT_FIELDS)
    row["schema_version"] = SCHEMA_VERSION
    row["timestamp"] = utc_timestamp()
    row["task_id"] = task_id
    row["run_index"] = run_index
    for field in dataclasses.fields(ctx):
        if field.name != "out":
            row[field.name] = getattr(ctx, field.name)
    row.update(measures)
    return row


def append_row(out: Path | str, row: dict[str, Any]) -> dict[str, Any]:
    """Append and durably sync one JSONL row, whatever its shape.

    Schema-neutral on purpose: the throughput instruments carry their own shapes and
    must not be forced through RESULT_FIELDS, but they still need the lock and the
    fsync -- a plain flush loses the row the reaper is about to make unrepeatable.
    """

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    with out.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return row


def emit(ctx: RunContext, task_id: str, run_index: int, **measures: Any) -> dict[str, Any]:
    """Append and durably sync exactly one fixed-shape JSONL result row."""

    return append_row(ctx.out, build_row(ctx, task_id, run_index, **measures))


def _metrics_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
    path = parsed.path
    if path.endswith("/v1"):
        path = path[:-3]
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path + "/metrics", "", ""))


# Monotonic counters, so a before/after pair around one task yields that task's
# share; the last two are gauges and are read as-is.
_METRIC_NAMES = {
    "llamacpp:prompt_tokens_total": "prompt_tokens_total",
    "llamacpp:prompt_tokens_cached_total": "prompt_tokens_cached_total",
    "llamacpp:prompt_seconds_total": "prompt_seconds_total",
    "llamacpp:tokens_predicted_total": "tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total": "tokens_predicted_seconds_total",
    "llamacpp:spec_decode_num_draft_tokens_total": "draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total": "draft_accepted_total",
    "llamacpp:spec_decode_num_drafts_total": "drafts_total",
    "llamacpp:n_tokens_max": "n_tokens_max",
    "llamacpp:prompt_tokens_seconds": "prefill_tps",
    "llamacpp:predicted_tokens_seconds": "decode_tps",
    # vLLM's equivalents, so a vLLM leg records the same eight measures rather than
    # nulls. Only one server type ever answers, so the two prefixes cannot collide.
    # Prefill volume must come from request_prefill_kv_computed_tokens_sum, not
    # prompt_tokens_by_source_total: the latter splits across a `source` label the
    # parser drops, so its three series would overwrite each other.
    "vllm:request_prefill_kv_computed_tokens_sum": "prompt_tokens_total",
    "vllm:prefix_cache_hits_total": "prompt_tokens_cached_total",
    "vllm:request_prefill_time_seconds_sum": "prompt_seconds_total",
    "vllm:generation_tokens_total": "tokens_predicted_total",
    "vllm:request_decode_time_seconds_sum": "tokens_predicted_seconds_total",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total": "draft_accepted_total",
    "vllm:spec_decode_num_drafts_total": "drafts_total",
}


def metrics_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Per-task telemetry from a counter pair. Empty if either scrape failed.

    A server restart resets the counters, so a negative delta means the pair does
    not describe one task and the whole reading is dropped rather than clamped.
    """

    if not before or not after:
        return {}
    pairs = {
        "prefill_tokens": "prompt_tokens_total",
        "cached_prompt_tokens": "prompt_tokens_cached_total",
        "prefill_seconds": "prompt_seconds_total",
        "decode_tokens": "tokens_predicted_total",
        "decode_seconds": "tokens_predicted_seconds_total",
        "draft_tokens": "draft_tokens_total",
        "draft_accepted_tokens": "draft_accepted_total",
        "drafts": "drafts_total",
    }
    out: dict[str, float] = {}
    for name, counter in pairs.items():
        if counter not in before or counter not in after:
            continue
        value = after[counter] - before[counter]
        if value < 0:
            return {}
        out[name] = value
    if "n_tokens_max" in after:
        # Process-wide high-water mark. Kept for the record, but per-task depth
        # comes from ContextSampler; this cannot be attributed to one task.
        out["server_high_water_tokens"] = after["n_tokens_max"]
    if out.get("prefill_seconds"):
        out["prefill_tps"] = out["prefill_tokens"] / out["prefill_seconds"]
    if out.get("decode_seconds"):
        out["decode_tps"] = out["decode_tokens"] / out["decode_seconds"]
    return out


def _slots_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path + "/slots", "", ""))


class ContextSampler(threading.Thread):
    """Track the deepest context a task actually occupies.

    llama-server's `n_tokens_max` metric is a process-wide high-water mark, so it
    cannot attribute depth to a task. `/slots` reports the live request instead:
    prompt length plus tokens decoded so far is the true occupancy.
    """

    def __init__(self, base_url: str, interval: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.url = _slots_url(base_url)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _sample(self) -> None:
        with urllib.request.urlopen(self.url, timeout=3) as response:
            slots = json.loads(response.read())
        for slot in slots if isinstance(slots, list) else [slots]:
            if not isinstance(slot, dict) or not slot.get("is_processing"):
                continue
            decoded = 0
            nxt = slot.get("next_token")
            if isinstance(nxt, list) and nxt and isinstance(nxt[0], dict):
                decoded = nxt[0].get("n_decoded") or 0
            depth = (slot.get("n_prompt_tokens") or 0) + decoded
            self.peak = max(self.peak, depth)

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:
                # Sampling is observational: never let it fail a task.
                continue

    def stop(self) -> int:
        self._stop.set()
        return self.peak


def _props_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path + "/props", "", ""))


def _models_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path + "/models", "", ""))


def _probe_openai_models(base_url: str, timeout: float) -> dict[str, object]:
    """Fallback for servers without llama.cpp's /props, which vLLM 404s.

    Without it a vLLM leg records no served_model_path at all, so the guard below
    is silently absent on exactly the legs whose checkpoint is a separate download.
    """

    try:
        with urllib.request.urlopen(_models_url(base_url), timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except (OSError, urllib.error.URLError, ValueError):
        return {}
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return {}
    entry = data[0]
    out: dict[str, object] = {}
    if isinstance(entry.get("root"), str):
        out["served_model_path"] = entry["root"]
    if isinstance(entry.get("max_model_len"), int):
        out["context_limit"] = entry["max_model_len"]
    return out


def server_alive(base_url: str, attempts: int = 3, timeout: float = 5.0) -> bool:
    """Is the endpoint still answering? Retried, because one blip must not end a leg.

    A leg that loses its server records every remaining task as a model failure with
    null token counts and still exits 0, and the box carrying the server log is torn
    down by the driver's own EXIT trap.
    """

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(_models_url(base_url), timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8", errors="replace"))
        except (OSError, urllib.error.URLError, ValueError):
            body = None
        if isinstance(body, dict) and isinstance(body.get("data"), list):
            return True
        if attempt + 1 < attempts:
            time.sleep(2.0)
    return False


def probe_server(base_url: str, timeout: float = 10.0) -> dict[str, object]:
    """Read the served config so every row is self-describing.

    Includes `served_model_path`, which is the only way to catch the silent
    catastrophe of this experiment: forgetting to swap the model between quants.
    """

    try:
        with urllib.request.urlopen(_props_url(base_url), timeout=timeout) as response:
            props = json.loads(response.read().decode("utf-8", errors="replace"))
    except (OSError, urllib.error.URLError, ValueError):
        return _probe_openai_models(base_url, timeout)
    if not isinstance(props, dict):
        return _probe_openai_models(base_url, timeout)
    gen = props.get("default_generation_settings") or {}
    params = gen.get("params") or {}
    out: dict[str, object] = {}
    if isinstance(gen.get("n_ctx"), int):
        out["context_limit"] = gen["n_ctx"]
    for key in ("temperature", "top_p", "top_k"):
        if key in params:
            out[key] = params[key]
    if isinstance(props.get("build_info"), str):
        out["engine_commit"] = props["build_info"]
    if isinstance(props.get("model_path"), str):
        out["served_model_path"] = props["model_path"]
    return out


def scrape_metrics(base_url: str, timeout: float = 5.0) -> dict[str, float]:
    """Return available llama-server counters and gauges, or an empty dict."""

    wanted = _METRIC_NAMES
    try:
        with urllib.request.urlopen(_metrics_url(base_url), timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, UnicodeError):
        return {}

    result: dict[str, float] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric = parts[0].split("{", 1)[0]
        output_name = wanted.get(metric)
        if output_name is None:
            continue
        try:
            result[output_name] = float(parts[1])
        except ValueError:
            continue
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class CodexStats:
    agent_turns: int | None
    tool_calls: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    # Cache reads as the *client* saw them. Unlike the llama-server counters this
    # is lane-independent, so it is the only cache figure comparable between a
    # local quant and an API model.
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


# C2 (control) is the direct competitor: what would be used instead of owning a GPU.
# C3 (frontier) is the ceiling: separates "quantization broke it" from "27B cannot do it".
LANE_MODELS = {
    "qwen": "qwen3.8-27b",
    "quant": "qwen3.8-27b",
    "control": "gpt-5.6-luna",
    "frontier": "gpt-5.6-sol",
    # The alternative the local card is actually competing with. Two tiers, because
    # which one is the fair comparison is the open question: on measured Polyglot
    # traffic the cheap tier costs about what renting the card does, and the frontier
    # tier costs five to ten times more, so "is local worth it" resolves entirely on
    # which of them this model's capability sits nearer.
    "deepseek-flash": "deepseek/deepseek-v4-flash",
    "deepseek-pro": "deepseek/deepseek-v4-pro",
    # Same cheap tier, two more routes. The vision SKU is 5.5x the input rate of plain
    # Flash and is here to price the multimodal option, not because Polyglot needs images.
    "deepseek-flash-vision": "deepseek/deepseek-v4-flash-vision-exp",
    # GLM's cheap tier, released this month. Its local path is closed -- no GGUF exists --
    # so the API is the only way to place it against the card at all.
    "glm-flash": "z-ai/glm-5.3-flash",
    # The same family served by someone else: qwen-flash is the cheap sibling, and
    # qwen-27b-api is this exact model at someone else's quantization -- fp8 on all but
    # one OpenRouter provider, so it reads as a light-quantization reference rather than
    # a full-precision control, and provider routing decides which.
    "qwen-flash": "qwen/qwen3.8-flash",
    "qwen-27b-api": "qwen/qwen3.8-27b",
    # The control lane again, but billed per token instead of against the Codex weekly
    # limit. Same model as `control`; the pair is what converts an opaque subscription
    # quota into a dollar figure.
    "luna-or": "openai/gpt-5.6-luna",
}

# The only lanes served by a card we rent, and so the only ones that owe a GPU capture.
LOCAL_LANES = {"qwen", "quant"}

# A slash means an OpenRouter model id rather than a first-party one; the provider has
# to be named or Codex routes it to its default and silently measures another model.
_ROUTED_LANES = {lane for lane, model in LANE_MODELS.items() if "/" in model}


def lane_model(lane: str) -> str:
    """Contract: the recorded model and the invoked model come from this one mapping."""

    try:
        return LANE_MODELS[lane]
    except KeyError:
        raise ValueError(
            "lane must be one of: " + ", ".join(sorted(LANE_MODELS))
        ) from None


# Per-process, because the local port is the only thing stopping two legs running at once:
# a profile pins a provider, a provider pins one base_url, and every leg tunnelled its box
# to that same port. A second profile on a second port is what makes legs parallel.
# Selected by environment rather than argument so it reaches `codex_command` and
# `codex_base_url` alike without threading a parameter through every call site -- and the
# endpoint guard in run_polyglot still compares what Codex will call against what is served,
# so a mismatched pair fails closed exactly as before.
CODEX_PROFILE = os.environ.get("QWEN38_CODEX_PROFILE", "qwen38")


def codex_base_url(profile: str = CODEX_PROFILE, home: Path | None = None) -> str | None:
    """The endpoint Codex will actually call, resolved from its own config.

    The local port is not the harness's to choose: the profile names a provider and the
    provider pins a base_url. A run that serves on some other port measures nothing, and
    a probe pointed at the port the harness picked will happily confirm a server Codex
    never talks to -- which is how a 40-task leg came back 0/40.

    Returns None when the config cannot be read, so the caller decides whether that is
    fatal; there is deliberately no guessed default.
    """

    home = home or Path.home() / ".codex"
    try:
        profile_cfg = tomllib.loads((home / f"{profile}.config.toml").read_text())
        provider = profile_cfg.get("model_provider")
        main = tomllib.loads((home / "config.toml").read_text())
        url = (main.get("model_providers", {}).get(provider, {}) or {}).get("base_url")
    except (OSError, ValueError, AttributeError):
        return None
    return url if isinstance(url, str) else None


def codex_command(lane: str, reasoning_effort: str) -> list[str]:
    """Build the single supported headless Codex command path for either lane."""

    command = [
        "codex",
        "exec",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        # Package fetches must work or a task with declared dependencies cannot be
        # compiled at all, and the agent codes blind. Set here rather than in the
        # user's config so every lane gets identical policy.
        "-c",
        "sandbox_workspace_write.network_access=true",
        # Compaction must fire below the served window or the server overflows. 110000
        # suits the 131072-token profile; a leg whose engine holds less must say so.
        "-c",
        f"model_auto_compact_token_limit={os.environ.get('QWEN38_AUTO_COMPACT_LIMIT', '110000')}",
    ]
    model = lane_model(lane)
    if lane in LOCAL_LANES:
        # Profile carries model, provider and context window; alias is uniform
        # across quants so the client cannot infer which one is loaded.
        command.extend(["-p", CODEX_PROFILE])
    else:
        command.extend(["-m", model])
        if lane in _ROUTED_LANES:
            command.extend(["-c", "model_provider=openrouter"])
    # Opt-in per leg, because the point of setting it is to compare against a leg that
    # did not. Codex already caps tool output near 12k tokens by default, so only a
    # value below that changes anything.
    cap = os.environ.get("QWEN38_TOOL_OUTPUT_TOKEN_LIMIT")
    if cap:
        command.extend(["-c", f"tool_output_token_limit={int(cap)}"])
    command.extend(["-c", f"model_reasoning_effort={reasoning_effort}", "--json", "-"])
    return command


def parse_codex_stats(output: str) -> CodexStats:
    """Parse Codex JSONL events; unknown values remain null rather than inferred."""

    saw_event = False
    saw_completed_turn = False
    turns = 0
    tool_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_input_tokens = 0
    cache_write_tokens = 0
    reasoning_tokens = 0
    saw_prompt_tokens = False
    saw_completion_tokens = False
    saw_cached = False
    saw_cache_write = False
    saw_reasoning = False
    tool_item_types = {"command_execution", "mcp_tool_call", "file_change"}

    for line in output.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            continue
        saw_event = True
        if event["type"] == "turn.completed":
            turns += 1
            saw_completed_turn = True
            usage = event.get("usage")
            if isinstance(usage, dict):
                value = usage.get("input_tokens")
                if isinstance(value, int):
                    prompt_tokens += value
                    saw_prompt_tokens = True
                value = usage.get("output_tokens")
                if isinstance(value, int):
                    completion_tokens += value
                    saw_completion_tokens = True
                # Codex and the Responses API have both spellings in the wild.
                value = usage.get("cached_input_tokens")
                if not isinstance(value, int):
                    details = usage.get("input_tokens_details")
                    if isinstance(details, dict):
                        value = details.get("cached_tokens")
                if isinstance(value, int):
                    cached_input_tokens += value
                    saw_cached = True
                value = usage.get("reasoning_output_tokens")
                if not isinstance(value, int):
                    details = usage.get("output_tokens_details")
                    if isinstance(details, dict):
                        value = details.get("reasoning_tokens")
                if isinstance(value, int):
                    reasoning_tokens += value
                    saw_reasoning = True
                value = usage.get("cache_write_input_tokens")
                if isinstance(value, int):
                    cache_write_tokens += value
                    saw_cache_write = True
        if event["type"] == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") in tool_item_types:
                tool_calls += 1

    return CodexStats(
        agent_turns=turns if saw_completed_turn else None,
        tool_calls=tool_calls if saw_event else None,
        prompt_tokens=prompt_tokens if saw_prompt_tokens else None,
        completion_tokens=completion_tokens if saw_completion_tokens else None,
        cached_input_tokens=cached_input_tokens if saw_cached else None,
        cache_write_tokens=cache_write_tokens if saw_cache_write else None,
        reasoning_tokens=reasoning_tokens if saw_reasoning else None,
    )


def invoke_codex(
    lane: str,
    reasoning_effort: str,
    prompt: str,
    cwd: Path,
    timeout: float,
    transcript: Path | None = None,
) -> tuple[CodexStats, float]:
    """Run one headless Codex turn and return parsed stats plus wall time.

    `codex exec` writes no rollout file, so the --json event stream on stdout is
    the only record of what was actually sent and returned. It is persisted
    before parsing, and on timeout or non-zero exit too -- a failed task is the
    one whose trace is most worth having.
    """

    started = time.monotonic()
    stdout = ""
    try:
        try:
            completed = subprocess.run(
                codex_command(lane, reasoning_effort),
                cwd=cwd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=True,
            )
            stdout = completed.stdout
        except subprocess.TimeoutExpired as expired:
            stdout = expired.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            raise
        except subprocess.CalledProcessError as failed:
            stdout = failed.stdout or ""
            raise
    finally:
        elapsed = time.monotonic() - started
        if transcript is not None:
            transcript.parent.mkdir(parents=True, exist_ok=True)
            with transcript.open("a", encoding="utf-8") as handle:
                # The prompt is the data-in half and Codex never echoes it, so it
                # is recorded here as an event of our own alongside the stream.
                handle.write(
                    json.dumps(
                        {"type": "harness.prompt", "lane": lane,
                         "reasoning_effort": reasoning_effort, "prompt": prompt},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.write(stdout)
                if stdout and not stdout.endswith("\n"):
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
    return parse_codex_stats(stdout), elapsed


ENV_FIELDS = ("gpu", "gpu_count", "gpu_devices", "gpu_driver", "cuda_toolkit", "power_limit_w")


def load_env(path: str | Path) -> dict[str, object]:
    """Read one scripts/gpuenv.py capture.

    Fails closed rather than recording null: a leg whose rows cannot say which card
    produced them is not worth the GPU time, and the null column is exactly the
    failure this channel exists to end.
    """

    try:
        env = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ENV CAPTURE UNREADABLE at {path}: {exc}") from None
    if not isinstance(env, dict):
        raise SystemExit(f"ENV CAPTURE MALFORMED at {path}: expected a JSON object")
    unknown = set(env) - set(ENV_FIELDS)
    if unknown:
        raise SystemExit(
            f"ENV CAPTURE has unknown field(s) {sorted(unknown)}; regenerate it with "
            "scripts/gpuenv.py rather than hand-editing"
        )
    if not env.get("gpu"):
        raise SystemExit(f"ENV CAPTURE at {path} names no GPU")
    return {k: v for k, v in env.items() if v is not None}


def apply_server_probe(ctx: RunContext, base_url: str) -> RunContext:
    """Fold the served config into the context, refusing a quant/model mismatch.

    Serving the previous quant after a failed swap is the one error that would
    corrupt every row while looking completely normal, so it fails the run.
    """

    probe = probe_server(base_url)
    if not probe:
        # Fail closed: an unreachable server means the quant guard never ran, so
        # the run could silently record the wrong model for every task.
        raise SystemExit(
            f"SERVER PROBE FAILED at {base_url}: cannot confirm which model is "
            "loaded. Start the server. There is deliberately no way to record "
            "blind: --metrics-url '' used to skip this guard, which is how three "
            "legs ran without it."
        )
    served = probe.get("served_model_path")
    if isinstance(served, str) and ctx.quant and ctx.quant not in served:
        raise SystemExit(
            f"QUANT MISMATCH: --quant {ctx.quant!r} but the server is running "
            f"{served!r}. Swap the model or fix the label; do not record this run."
        )
    return dataclasses.replace(ctx, **probe)


def _require_env(args: argparse.Namespace) -> dict[str, object]:
    """--dry-run is the no-GPU harness check, so it is the one local path exempt."""

    if getattr(args, "dry_run", False):
        return {}
    if not getattr(args, "env_json", None):
        raise SystemExit(
            "--env-json is required for a local lane: capture it with "
            "`ssh <box> python3 - < scripts/gpuenv.py > env.json`. Without it the rows "
            "cannot say which card produced them, which is how 192 rows of a "
            "GPU-purchase experiment ended up with a null gpu column."
        )
    return load_env(args.env_json)


def make_context(args: argparse.Namespace) -> RunContext:
    frontier = args.lane in {"frontier", "control"}
    hosted = args.lane not in LOCAL_LANES
    return RunContext(
        out=args.out,
        experiment_id=args.experiment_id,
        lane=args.lane,
        quant=args.quant,
        engine=None if frontier else args.engine,
        model=lane_model(args.lane),
        reasoning_effort=args.reasoning_effort,
        **({} if hosted else _require_env(args)),
    )


def add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lane",
        required=True,
        # Derived, not restated: a lane added to LANE_MODELS but missing here parses as an
        # error at launch, which is how the deepseek lanes stayed unrunnable after being
        # wired and unit-tested.
        choices=tuple(sorted(LANE_MODELS)),
        help="qwen/quant uses profile qwen38; control uses gpt-5.6-luna; frontier uses "
        "gpt-5.6-sol; slash-bearing ids route through OpenRouter",
    )
    parser.add_argument("--quant", required=True, help="quantization label recorded verbatim")
    parser.add_argument(
        "--engine",
        default="llama.cpp",
        help="serving engine recorded verbatim; the default is only correct because "
        "every local lane predating this flag served llama-server -- a vLLM or ninfer "
        "run that omits it writes rows labelled llama.cpp",
    )
    parser.add_argument("--run-index", required=True, type=int, help="paired repetition index")
    parser.add_argument("--out", required=True, type=Path, help="append-only result JSONL path")
    parser.add_argument("--experiment-id", help="optional experiment identifier")
    parser.add_argument(
        "--env-json",
        help="scripts/gpuenv.py capture describing the GPU this leg ran on. Required "
        "for local lanes: capture it per leg, after setting any power cap, since the "
        "cap is mutable and is itself an experiment variable",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="xhigh",
        help="Codex model_reasoning_effort; the Qwen template accepts only "
        "xhigh|medium|low, silently aliases high->xhigh, and raises on minimal",
    )


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Append one durable fixed-shape evaluation result row."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument("--lane", default="qwen")
    parser.add_argument("--quant", default="smoke")
    args = parser.parse_args()
    emit(
        RunContext(out=args.out, lane=args.lane, quant=args.quant),
        args.task_id,
        args.run_index,
        notes="emit.py smoke test",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
