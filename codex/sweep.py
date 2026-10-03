#!/usr/bin/env python3
"""Plan gpuq manifests without launching processes; execution is parent-only."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import time
import tomllib

import gpu_harness as h

ROOT = Path(__file__).resolve().parent.parent


def load_matrix(path):
    cfg = tomllib.loads(path.read_text())
    arms = cfg["arms"]
    names = [arm["name"] for arm in arms]
    if len(names) != len(set(names)) or len(names) < 2 or any(not re.fullmatch(r"[a-z0-9-]+", name) for name in names):
        raise ValueError("matrix needs unique safe arm names")
    if cfg["baseline"] != names[0] or cfg["canary_arm"] not in names:
        raise ValueError("baseline must be first; canary arm must exist")
    for arm in arms:
        flags = arm["flags"]
        if any(not isinstance(flag, str) or not flag for flag in flags) or len(flags) % 2:
            raise ValueError("arm flags must be option/value pairs")
        pairs = dict(zip(flags[::2], flags[1::2]))
        if len(pairs) != len(flags) // 2 or any(not key.startswith("--") for key in pairs):
            raise ValueError("arm flags need distinct long options")
        if set(pairs) & {"--model", "--host", "--port", "--parallel", "--ctx-size", "--spec-draft-n-min"}:
            raise ValueError("arm cannot override harness ownership, depth or n_min")
        k = int(pairs.get("--spec-draft-n-max", "5"))
        p_min = float(pairs.get("--spec-draft-p-min", "0"))
        if not 1 <= k <= 32 or not 0 <= p_min <= 1:
            raise ValueError("invalid draft settings")
        arm["k"] = k
    if cfg["rounds"] < 40 or not 8 <= cfg["followups"] <= 16 or cfg["depths"] != [8192, 90000]:
        raise ValueError("F1 needs >=40 rounds, 8..16 followups and the two declared depths")
    return cfg


def order(cfg):
    names = [arm["name"] for arm in cfg["arms"]]
    return names + names[::-1]


def costs(cfg):
    c = cfg["cost"]
    common = c["boot_s_G"] + c["warmup_cold_s_G"] + c["prefix_calibration_s_G"] + c["stop_hash_s_G"]
    per_boot = common + c["shallow_request_s_M"] * 3 + c["deep_cold_s_D"] + cfg["followups"] * c["followup_s_G"]
    per_arm = {}
    for arm in cfg["arms"]:
        extra = 4 * c["generation_s_M"] * max(0, arm["k"] / 5 - 1)
        # Larger k gets a pessimistic linear generation allowance until that arm has a recorded boot.
        per_arm[arm["name"]] = arm.get("cost", {}).get("per_boot_s_M", per_boot + extra)
    if any(not finite(value) or value <= 0 for value in per_arm.values()):
        raise ValueError("invalid per-arm boot cost")
    canary = per_arm[cfg["canary_arm"]] - (cfg["followups"] - 2) * c["followup_s_G"]
    estimate = math.ceil(2 * sum(per_arm.values()) + c["initial_hash_s_G"])
    canary_estimate = math.ceil(canary + c["initial_hash_s_G"])
    budget = math.ceil(estimate / 0.7 / 60) * 60
    if canary_estimate > 0.7 * 600 or budget + 600 > cfg["max_total_budget_s"]:
        raise ValueError("stage estimates exceed 70% headroom or total GPU budget")
    return {"full_est_s_D": estimate, "full_budget_s_D": budget, "canary_est_s_D": canary_estimate,
            "canary_budget_s_D": 600, "total_budget_s_D": budget + 600, "per_arm_boot_s_D": per_arm,
            "stage_inputs": c}


def stock_files(cfg):
    root = Path(cfg["frozen"]).expanduser().resolve()
    directory = root / "build/bin"
    manifest = directory / "SHA256SUMS"
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or Path(name).name != name or name in expected:
            raise ValueError("invalid stock frozen manifest")
        expected[name] = digest
    if set(expected) != {p.name for p in directory.iterdir() if p.name != "SHA256SUMS"} or "weekend-srv" not in expected:
        raise ValueError("stock frozen file set changed")
    if (root / "SOURCE").read_text().splitlines()[0] != cfg["source"]:
        raise ValueError("stock frozen SOURCE changed")
    for name in expected:
        if not (directory / name).resolve().is_relative_to(directory):
            raise ValueError("frozen symlink leaves binary directory")
    return root, directory, expected


def pins(cfg, matrix):
    root, directory, expected = stock_files(cfg)
    return sorted({matrix.resolve(), Path(__file__).resolve(), ROOT / "codex/gpu_harness.py", ROOT / "codex/gpu_ab.sh",
                   root / "SOURCE", directory / "SHA256SUMS", *[directory / name for name in expected],
                   Path(cfg["model"]).expanduser().resolve(), Path(cfg["vocab"]).expanduser().resolve(),
                   *h.INPUTS.glob("*.json"), h.INPUTS / "chat-template.jinja"})


def plan(cfg, matrix, output):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("plan output must stay inside this worktree")
    output.mkdir(parents=True, exist_ok=True)
    cost = costs(cfg)
    j = json.dumps
    base = ROOT / "codex/runs" / cfg["id"]
    lines = [f'id = {j(cfg["id"])}', 'title = "Swift stack2 S5: top-q early stop x maximum MTP draft length"',
             'kind = "measure"', 'ab = true', 'retry = "fresh"', f'budget_s = {cost["total_budget_s_D"]}',
             f'est_s = {cost["full_est_s_D"] + cost["canary_est_s_D"]}', 'startup_grace_s = 180',
             'signatures = [',
             '  { class = "hazard", regex = "\\\\bXid\\\\b|fallen off the bus" },',
             '  { class = "timeout", regex = "queue wall budget" },',
             '  { class = "integrity", regex = "frozen.*changed|inputs changed|model SHA changed|identity failed" },',
             '  { class = "insufficient_data", regex = "insufficient steady|missing rounds|counter coverage" },',
             '  { class = "deterministic", regex = "boot validation failed" },', ']']
    for stage in ("run", "preflight", "canary"):
        cmd = ["bash", str(ROOT / "codex/gpu_ab.sh"), "run", str(matrix.resolve())]
        if stage != "preflight":
            cmd += ["--execute", "--port", "{port}", "--out", str(base / (stage + "-a{attempt}"))]
        if stage == "canary":
            cmd += ["--canary"]
        lines += ["", f"[{stage}]", f"cmd = {j(cmd)}", f"cwd = {j(str(ROOT))}",
                  'env = { RUN_GPU = "1" }' if stage != "preflight" else 'env = {}']
        if stage == "run":
            lines += ['ports = "dynamic"']
        if stage != "run":
            lines += ["timeout_s = 600"]
    lines += ["", "[outputs]", f"paths = {j([str(base)])}", "", "[pins]", f"files = {j([str(p) for p in pins(cfg, matrix)])}",
              "", "[fix]", "editable = []"]
    manifest = output / (cfg["id"] + ".toml")
    manifest.write_text("\n".join(lines) + "\n")
    (output / (cfg["id"] + "-plan.json")).write_text(j({"boots": order(cfg), "canary_arm": cfg["canary_arm"], **cost}, indent=2) + "\n")
    return manifest, cost


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def counters(record):
    m, t = record["metrics"], record["response"]["timings"]
    if any(not finite(m.get(key)) or m[key] != int(m[key]) for key in ("drafts", "draft_tokens", "accepted_tokens")):
        raise ValueError("invalid authoritative counters")
    if m["draft_tokens"] != t.get("draft_n", 0) or m["accepted_tokens"] != t.get("draft_n_accepted", 0) or not 0 <= m["accepted_tokens"] <= m["draft_tokens"]:
        raise ValueError("authoritative counters disagree with final timings")
    return m, t


def trace_rounds(text, metrics):
    total = accepted = rounds = 0
    replay = False
    mapping = {}
    for line in text.splitlines():
        match = re.search(r"accepted\s+(\d+)/\s*(\d+) draft tokens( \(restore checkpoint\))?$", line)
        if not match:
            continue
        a, d = map(int, match.group(1, 2))
        if not 0 <= a <= d or d == 0:
            raise ValueError("invalid trace round")
        total += d
        if match.group(3):
            replay = True
            continue
        if replay and a > 0:
            a -= 1
        replay = False
        accepted += a
        rounds += 1
        mapping[total] = rounds
    if (rounds, total, accepted) != (metrics["drafts"], metrics["draft_tokens"], metrics["accepted_tokens"]):
        raise ValueError("counter coverage: trace totals disagree with /metrics")
    return mapping


def fixed_rounds(metrics, k):
    rounds, tokens = int(metrics["drafts"]), int(metrics["draft_tokens"])
    if rounds <= 0 or not 0 <= k * rounds - tokens <= k * (k - 1) // 2:
        raise ValueError("fixed-K reconstruction requires full drafts before the tail")
    # Recorded fixed-K captures have no round trace. Only full-K interior counters are used.
    return {k * r: r for r in range(1, tokens // k + 1)}


def steady(record, k, minimum):
    m, final = counters(record)
    mapping = trace_rounds(record["round_log"], m) if "round_log" in record else fixed_rounds(m, k)
    points = []
    previous = (0, 0, 0, 0)
    for chunk in record["chunks"]:
        t = chunk.get("timings") or {}
        if any(not finite(t.get(key)) for key in ("predicted_n", "predicted_ms")):
            continue
        if "draft_n" not in t and m["draft_tokens"] == 0:
            n = a = 0
        else:
            n, a = t.get("draft_n"), t.get("draft_n_accepted")
        if not finite(n) or not finite(a):
            continue
        n, a, ms, out = n, a, t["predicted_ms"], t["predicted_n"]
        old_n, old_a, old_ms, old_out = previous
        if n < old_n or a < old_a or ms < old_ms or out < old_out or n > m["draft_tokens"] or a > m["accepted_tokens"] or ms > final["predicted_ms"]:
            raise ValueError("reversed or inconsistent cumulative SSE timing")
        final_only = not chunk.get("choices") or all(choice.get("finish_reason") for choice in chunk["choices"])
        if ms > old_ms and not final_only:
            points.append({"ms": ms, "drafted": n, "accepted": a, "out": out})
        elif ms == old_ms:
            if n != old_n or a != old_a:
                raise ValueError("distinct rounds share a timing stamp")
            if points and points[-1]["ms"] == ms:
                points[-1]["out"] = out
        previous = n, a, ms, out
    selected = []
    if m["draft_tokens"] == 0:
        for index, (left, right) in enumerate(zip(points, points[1:]), 1):
            if index >= 3 and right["out"] <= final["predicted_n"] - k - 1 and right["out"] - left["out"] == 1:
                selected.append({"ms_M": right["ms"] - left["ms"], "drafted_M": 0, "accepted_M": 0, "emitted_M": 1})
    else:
        # Zero-proposal decodes between two verifications belong to that interval. Do not count their stamps twice.
        groups = []
        for point in points:
            if point["drafted"] == 0:
                continue
            if groups and point["drafted"] == groups[-1]["drafted"]:
                continue
            groups.append(point)
        for left, right in zip(groups, groups[1:]):
            lround, rround = mapping.get(left["drafted"]), mapping.get(right["drafted"])
            if lround is None or rround is None or lround < 3 or rround != lround + 1 or right["out"] > final["predicted_n"] - k - 1:
                continue
            # First stamps delimit the verification cycle; accepted+1 is exact even when the tool parser hides token chunks.
            selected.append({"ms_M": right["ms"] - left["ms"], "drafted_M": right["drafted"] - left["drafted"],
                             "accepted_M": right["accepted"] - left["accepted"], "emitted_M": right["accepted"] - left["accepted"] + 1})
    if len(selected) < minimum:
        raise ValueError(f"insufficient steady rounds: {len(selected)} < {minimum}")
    ms = statistics.mean(row["ms_M"] for row in selected)
    drafted = sum(row["drafted_M"] for row in selected)
    accepted = sum(row["accepted_M"] for row in selected)
    return {"steady_rounds_M": len(selected), "steady_ms_per_round_D": ms,
            "steady_sd_ms_M": statistics.pstdev(row["ms_M"] for row in selected),
            "drafted_per_round_D": drafted / len(selected), "acceptance_D": accepted / drafted if drafted else None,
            "verified_tokens_per_round_D": statistics.mean(row["emitted_M"] for row in selected),
            "all_zero_proposal": drafted == 0, "intervals": selected}


def tool_gate(record, long=False):
    choices = record["response"]["choices"]
    calls = choices[0]["message"].get("tool_calls", [])
    if long and not any(call["function"]["name"] == "write_file" and call["function"]["arguments"] for call in calls):
        raise ValueError("long request did not stream visible write_file arguments")
    for call in calls:
        arguments = call["function"]["arguments"]
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            # Long fixtures deliberately stream a file larger than the completion cap. They are never replayed.
            if long and choices[0].get("finish_reason") == "length" and arguments.startswith('{"content":'):
                continue
            raise ValueError("cut tool call arguments")
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be an object")
    if not long and calls and choices[0].get("finish_reason") == "length":
        raise ValueError("tool call hit max_tokens")


def validate(results, dest, arm, cfg):
    failures = {}
    for case, record in results.items():
        try:
            if "error" in record:
                raise ValueError(record["error"]["message"])
            m, t = counters(record)
            if any(not finite(t.get(key)) for key in ("predicted_n", "predicted_ms", "prompt_ms", "prompt_n", "cache_n")):
                raise ValueError("invalid timing or depth")
            if t["predicted_n"] >= 32768:
                raise ValueError("runaway response")
            long = case.startswith("long-")
            tool_gate(record, long)
            if case == "cold-tool" and t["cache_n"] != 0:
                raise ValueError("cold request reused a prompt cache")
            if long:
                record["steady"] = steady(record, arm["k"], cfg["rounds"])
                if t["predicted_n"] < 2048:
                    raise ValueError("long request generated fewer than 2048 tokens")
                depth = 90000 if "d90000" in case else 8192
                if abs(t["prompt_n"] + t["cache_n"] - depth) > 1024:
                    raise ValueError("long request missed declared context depth")
                if "d8192" in case or "tool-d90000" in case:
                    if t["cache_n"] != 0:
                        raise ValueError("cold long request reused cache")
                elif t["cache_n"] < 79000:
                    raise ValueError("deep prefix was not retained")
            if case.startswith("short-tool-") and case != "short-tool-1" and t["cache_n"] <= 0:
                raise ValueError("agent follow-up did not reuse cache")
        except (ValueError, KeyError, TypeError) as error:
            failures[case] = str(error)
    (dest / "boot-validation.json").write_text(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    return failures


def server_gate(text):
    if not re.search(r"n_slots = 1, n_ctx_slot = 102400", text) or "falling back to internal AllReduce" not in text:
        raise ValueError("server did not confirm np1, context or internal AllReduce")
    if re.search(r"backend offload failed|using CPU sampler|out of memory|CUDA error|illegal memory access", text, re.I):
        raise ValueError("server backend failure or sampling fallback")


def canary_gate(results):
    if not any(record.get("metrics", {}).get("drafts", 0) > 0 for case, record in results.items() if case.startswith("long-")):
        raise ValueError("canary did not exercise nonempty verification trace")


def identity(record):
    choices = json.loads(json.dumps(record["response"]["choices"]))
    for choice in choices:
        for call in choice["message"].get("tool_calls", []):
            call.pop("id", None)
    t = record["response"]["timings"]
    return choices, record["metrics"], {key: t[key] for key in ("predicted_n", "prompt_n", "cache_n")}


def repeat_gate(boots, cfg):
    for arm in cfg["arms"]:
        pair = [boot["requests"] for boot in boots if boot["arm"] == arm["name"]]
        if len(pair) != 2 or pair[0].keys() != pair[1].keys():
            raise ValueError("unbalanced arm boots")
        for case in pair[0]:
            if identity(pair[0][case]) != identity(pair[1][case]):
                raise ValueError("same-arm identity failed: " + arm["name"] + "/" + case)


class MatrixHarness(h.Harness):
    def __init__(self, args, out):
        super().__init__(args, out)
        self.env["LLAMA_TRACE"] = "1"

    def request(self, dest, *args, **kwargs):
        log = dest / "server.log"
        start = log.stat().st_size
        record = super().request(dest, *args, **kwargs)
        until = time.monotonic() + self.timeout(limit=1)
        while True:
            with log.open("rb") as stream:
                stream.seek(start)
                record["round_log"] = stream.read().decode(errors="replace")
            if "stop processing:" in record["round_log"] or "error" in record or time.monotonic() >= until:
                break
            time.sleep(0.01)
        case, number = args[:2]
        (dest / f"{case}-{number}.json").write_text(json.dumps(record, indent=2))
        return record

    def configure_command(self, cmd):
        for option, value in zip(self.arm["flags"][::2], self.arm["flags"][1::2]):
            if option in cmd:
                cmd[cmd.index(option) + 1] = value
            else:
                cmd += [option, value]
        return cmd

    def deep_body(self, body, target):
        body = json.loads(json.dumps(body))
        original = body["messages"][0]["content"]
        material = "\n// Archived parser context.\ndef parse_records(lines):\n    return [line.split(':', 1) for line in lines if line.strip()]\n"
        ids = self.http("/tokenize", {"content": material * 7000})["tokens"]
        count = target - 8400
        for _ in range(3):
            prefix = self.http("/detokenize", {"tokens": ids[:count]})["content"]
            body["messages"][0]["content"] = original + "\nReference archive, for context only:\n" + prefix
            prompt = self.http("/apply-template", body)["prompt"]
            measured = len(self.http("/tokenize", {"content": prompt, "add_special": True})["tokens"])
            if abs(measured - target) <= 64:
                return body
            count += target - measured
            if not 0 < count < len(ids):
                raise ValueError("prefix token calibration out of range")
        raise ValueError("prefix token calibration did not converge")


def suite(harness, dest, cfg):
    cold = json.loads((h.INPUTS / "tool.json").read_text())
    results = {"cold-tool": harness.request(dest, "cold-tool", 1, cold)}
    deep = None
    prefix_error = None
    for depth in cfg["depths"]:
        for kind in ("tool", "think"):
            body = json.loads((h.INPUTS / f"long-{kind}.json").read_text())
            if depth == 90000:
                if deep is None and prefix_error is None:
                    try:
                        deep = harness.deep_body(body, depth)
                    except (OSError, ValueError, RuntimeError) as error:
                        prefix_error = {"type": type(error).__name__, "message": str(error)}
                        if isinstance(error, h.urllib.error.HTTPError):
                            prefix_error["body"] = error.read().decode(errors="replace")
                        (dest / "prefix-error.json").write_text(json.dumps(prefix_error, indent=2))
                if prefix_error:
                    label = f"long-{kind}-d{depth}"
                    results[label] = {"case": label, "error": prefix_error}
                    (dest / (label + "-1.json")).write_text(json.dumps(results[label], indent=2))
                    continue
                body["messages"] = deep["messages"]
                body["cache_prompt"] = kind == "think"
            label = f"long-{kind}-d{depth}"
            results[label] = harness.request(dest, label, 1, body, stream=True)
    body = json.loads((h.INPUTS / "tool.json").read_text())
    body.update(cache_prompt=True, max_tokens=768)
    for number in range(1, (2 if harness.args.canary else cfg["followups"]) + 1):
        label = f"short-tool-{number}"
        record = harness.request(dest, label, 1, body)
        results[label] = record
        if "error" in record:
            break
        try:
            tool_gate(record)
        except ValueError:
            break
        message = json.loads(json.dumps(record["response"]["choices"][0]["message"]))
        for index, call in enumerate(message.get("tool_calls", [])):
            call["id"] = f"followup_{number}_{index}"
        body["messages"].append(message)
        for call in message.get("tool_calls", []):
            body["messages"].append({"role": "tool", "tool_call_id": call["id"], "content": "Fixture result: command not executed. Continue with the next step."})
        body["messages"].append({"role": "user", "content": "Continue fixing the parser. Give the next small tool call."})
    return results


def summarize(boots, cfg):
    rows = []
    for boot in boots:
        result = boot["requests"]
        reference = statistics.mean(record["steady"]["steady_ms_per_round_D"] for name, record in result.items() if name.startswith("long-") and "d8192" in name)
        for case, record in result.items():
            m, t = counters(record)
            wall = (record["end_ns"] - record["begin_ns"]) / 1e6
            row = {"arm": boot["arm"], "boot": boot["boot"], "case": case, "wall_ms_M": wall,
                   "prompt_ms_M": t["prompt_ms"], "generation_ms_M": t["predicted_ms"],
                   "depth_tokens_M": t["prompt_n"] + t["cache_n"], "cache_tokens_M": t["cache_n"],
                   "drafted_M": m["draft_tokens"], "accepted_M": m["accepted_tokens"], "nonempty_rounds_M": m["drafts"],
                   "tokens_per_round_D": t["predicted_n"] / m["drafts"] if m["drafts"] else 1,
                   "acceptance_D": m["accepted_tokens"] / m["draft_tokens"] if m["draft_tokens"] else None,
                   "emitted_tokens_s_D": 1000 * t["predicted_n"] / t["predicted_ms"],
                   "wall_tokens_s_D": 1000 * t["predicted_n"] / wall,
                   "outside_prompt_generation_ms_D": wall - t["prompt_ms"] - t["predicted_ms"]}
            if "steady" in record:
                row.update({key: value for key, value in record["steady"].items() if key != "intervals"})
                row["verified_tokens_s_proxy_D"] = 1000 * row["verified_tokens_per_round_D"] / row["steady_ms_per_round_D"]
                cycles = t["predicted_n"] - 1 if record["steady"]["all_zero_proposal"] else m["drafts"]
                row["request_fixed_ms_estimate_D"] = wall - t["prompt_ms"] - cycles * row["steady_ms_per_round_D"]
            else:
                row["reference_steady_ms_per_round_D"] = reference
                row["request_fixed_ms_estimate_D"] = wall - t["prompt_ms"] - m["drafts"] * reference
            rows.append(row)
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    paired = []
    per_arm = []
    for arm in cfg["arms"]:
        for case in boots[0]["requests"]:
            chosen = [row for row in rows if row["arm"] == arm["name"] and row["case"] == case]
            keys = [key for key in chosen[0] if key.endswith(("_M", "_D")) and all(number(row.get(key)) for row in chosen)]
            per_arm.append({"arm": arm["name"], "case": case, "boots_M": len(chosen),
                            "mean_D": {key: statistics.mean(row[key] for row in chosen) for key in keys},
                            "boot_range_M": {key: [min(row[key] for row in chosen), max(row[key] for row in chosen)] for key in keys}})
    for arm in cfg["arms"][1:]:
        for case in boots[0]["requests"]:
            base = [row for row in rows if row["arm"] == cfg["baseline"] and row["case"] == case]
            chosen = [row for row in rows if row["arm"] == arm["name"] and row["case"] == case]
            if len(base) != 2 or len(chosen) != 2:
                raise ValueError("unbalanced arm boots")
            keys = [key for key in base[0] if key.endswith(("_M", "_D")) and all(number(row.get(key)) for row in base + chosen)]
            differences = [{key: row[key] - ref[key] for key in keys} for ref, row in zip(base, chosen)]
            paired.append({"arm": arm["name"], "case": case, "differences_D": differences,
                           "mean_difference_D": {key: statistics.mean(pair[key] for pair in differences) for key in keys},
                           "difference_range_D": {key: [min(pair[key] for pair in differences), max(pair[key] for pair in differences)] for key in keys}})
    return {"requests": rows, "paired": paired, "per_arm": per_arm,
            "round_definition": "Adjacent trace-identified nonempty verifications, after three rounds and before the last k+1 output tokens. SSE clock differences include intervening zero-proposal work. All-zero arms use adjacent one-token SSE clocks.",
            "fixed_cost_definition": "wall - prompt - nonempty_rounds * steady_ms_per_round (all-zero long: output-1 rounds). Short requests use the shallow steady reference. Estimates include ramp/tail and are not pure dispatch cost. tokens_per_round is whole-generation output/nonempty verifications; verified_tokens_per_round is accepted+1 for selected intervals. verified_tokens_s_proxy excludes zero-proposal emissions. emitted_tokens_s uses authoritative whole-generation output and time.",
            "pairing": "First occurrence with first baseline; second with second baseline. Mirrored positions balance drift; ranges are boot spread, not confidence intervals."}


def snapshot(cfg, out):
    root, directory, expected = stock_files(cfg)
    dest = out / "frozen"
    (dest / "bin").mkdir(parents=True)
    for name, digest in expected.items():
        if h.sha(directory / name) != digest:
            raise ValueError("stock frozen bytes changed: " + name)
        if name == "llama-server":
            continue
        target = "sweep-srv" if name == "weekend-srv" else name
        shutil.copy2(directory / name, dest / "bin" / target)
    shutil.copy2(root / "SOURCE", dest / "SOURCE")
    (dest / "bin/SHA256SUMS").write_text("".join(f"{digest}  {'sweep-srv' if name == 'weekend-srv' else name}\n" for name, digest in expected.items() if name != "llama-server"))
    (dest / "SHA256SUMS").write_text("".join(f"{h.sha(dest / name)}  {name}\n" for name in ("SOURCE", "bin/SHA256SUMS")))
    h.frozen(dest)
    return dest


def execute(args, cfg):
    if os.environ.get("RUN_GPU") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("parent execution needs RUN_GPU=1 and selected CUDA_VISIBLE_DEVICES")
    if not os.environ.get("XID_SINCE"):
        raise RuntimeError("parent execution requires XID_SINCE")
    if not args.out or not args.out.resolve().is_relative_to(ROOT):
        raise ValueError("execution output must stay inside this worktree")
    if not 0 < args.port < 65536:
        raise ValueError("invalid server port")
    args.out.mkdir(parents=True, exist_ok=False)
    cost = costs(cfg)
    args.budget = 600 if args.canary else cost["full_budget_s_D"]
    harness = MatrixHarness(args, args.out)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    boots = []
    try:
        inventory = pins(cfg, args.matrix)
        hashes = {str(path): h.sha(path) for path in inventory}
        if hashes[str(Path(cfg["model"]).expanduser().resolve())] != cfg["model_sha256"]:
            raise ValueError("model SHA changed")
        (args.out / "provenance.json").write_text(json.dumps({"hashes": hashes, "config": cfg, "costs": cost}, indent=2))
        frozen = snapshot(cfg, args.out)
        names = [cfg["canary_arm"]] if args.canary else order(cfg)
        for index, name in enumerate(names, 1):
            harness.arm = next(arm for arm in cfg["arms"] if arm["name"] == name)
            args.k = harness.arm["k"]
            dest = harness.boot(frozen, f"{index}-{name}")
            results = suite(harness, dest, cfg)
            failures = validate(results, dest, harness.arm, cfg)
            if args.canary:
                try:
                    canary_gate(results)
                except ValueError as error:
                    failures["canary"] = str(error)
            try:
                server_gate((dest / "server.log").read_text())
            except ValueError as error:
                failures["server"] = str(error)
            expected_count = 7 if args.canary else 5 + cfg["followups"]
            if len(results) != expected_count:
                failures["suite"] = "follow-up set ended before every request was recorded"
            boots.append({"boot": index, "arm": name, "requests": results})
            (args.out / "results.json").write_text(json.dumps(boots, indent=2))
            harness.stop()
            harness.xid()
            h.frozen(dest)
            if failures:
                raise ValueError("boot validation failed after writing results: " + json.dumps(failures))
            print(f"M: boot={index} arm={name} requests={len(results)} gates passed", flush=True)
        if hashes != {path: h.sha(Path(path)) for path in hashes}:
            raise ValueError("inputs changed during run")
        harness.timeout(limit=1, reserve=0)
        if not args.canary:
            repeat_gate(boots, cfg)
        summary = {"canary": True, "requests": list(boots[0]["requests"])} if args.canary else summarize(boots, cfg)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        (args.out / "verdict.json").write_text(json.dumps({"verdict": "PASS", "canary": args.canary}))
    except BaseException as error:
        message = str(error)
        verdict = "hazard" if re.search(r"\bXid\b|fallen off the bus", message, re.I) else "budget" if "budget" in message else "INVALID"
        (args.out / "verdict.json").write_text(json.dumps({"verdict": verdict, "error": message}))
        raise
    finally:
        harness.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "codex/items")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--port", type=int, default=8398)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = load_matrix(args.matrix)
    h.MODEL, h.VOCAB = Path(cfg["model"]).expanduser(), Path(cfg["vocab"]).expanduser()
    if args.mode == "plan":
        if args.execute:
            parser.error("plan cannot execute")
        manifest, cost = plan(cfg, args.matrix, args.output)
        print(json.dumps({"manifest": str(manifest), **cost}, indent=2))
    elif args.execute:
        execute(args, cfg)
    else:
        stock_files(cfg)
        print(json.dumps({"boots": [cfg["canary_arm"]] if args.canary else order(cfg), **costs(cfg)}, indent=2))


if __name__ == "__main__":
    main()
