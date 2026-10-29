#!/usr/bin/env python3
"""Matched MTP-on/off throughput bench.

Speculative decoding changes speed, not answer quality, so it is measured on its
own rather than switched on mid-experiment: a paired quant contrast that mixed two
decoding algorithms would attribute the difference to the quant.

Draft acceptance depends on what is being generated, so the prompts are real task
prompts, and depth is swept because a draft head that pays at 8K need not pay at 64K.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from scripts import ROOT
from scripts.instruments import emit, longctx



def load_tasks(manifest: Path, count: int) -> list[dict]:
    data = json.loads(manifest.read_text())
    tasks = data["tasks"] if isinstance(data, dict) else data
    return sorted(tasks, key=lambda t: t["task_id"])[:count]


def bench_prompt(task: dict, manifest_dir: Path) -> str:
    """A tool-free restatement of the task. The agentic prompt cannot be reused:
    with no working directory the model would answer that it cannot see the files,
    and short refusals measure nothing. Stub plus tests forces real code generation,
    which is what draft acceptance has to be measured on."""

    base = (manifest_dir / task["source_dir"]).resolve()
    skip = {".meta", "node_modules", "target", ".git"}
    tests = sorted(
        str(path.relative_to(base))
        for path in base.rglob("*")
        if path.is_file()
        and not skip & set(path.relative_to(base).parts)
        and re.search(r"test|spec", str(path.relative_to(base)), re.IGNORECASE)
    )
    parts = [f"Implement this {task['language']} exercise. Reply with the complete solution file."]
    for name in list(task["solution_path"]) + tests:
        path = base / name
        if path.is_file():
            parts.append(f"--- {name} ---\n{path.read_text(errors='replace')}")
    return "\n\n".join(parts)


def generate(base_url: str, prompt: str, max_tokens: int, effort: str, timeout: float) -> dict:
    payload = {
        "model": emit.lane_model("quant"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def acceptance(row: dict) -> dict:
    """Per-draft acceptance is the diagnostic: a high token count with a low
    accepted/draft ratio means the head is drafting but guessing wrong."""
    drafts = row.get("drafts") or 0
    drafted = row.get("draft_tokens") or 0
    accepted = row.get("draft_accepted_tokens") or 0
    out = {}
    if drafted:
        out["draft_accept_rate"] = accepted / drafted
    if drafts:
        out["accepted_per_draft"] = accepted / drafts
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--label", required=True, help="mtp-off | mtp-on")
    parser.add_argument("--quant", required=True)
    parser.add_argument("--env-json", required=True,
                        help="scripts/gpuenv.py capture; without it these rows cannot "
                             "say which card produced the speculation numbers")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "subsets" / "polyglot.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--depths", default="0,64000")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    env = emit.load_env(args.env_json)
    probe = emit.probe_server(args.base_url)
    if args.quant not in str(probe.get("served_model_path", "")):
        raise SystemExit(f"server is not serving {args.quant}: {probe.get('served_model_path')}")

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    tasks = load_tasks(args.manifest, args.tasks)

    for depth in depths:
        for task in tasks:
            prompt = bench_prompt(task, args.manifest.parent)
            actual = 0
            if depth:
                try:
                    padding, actual = longctx.build_padding(task, depth, args.base_url)
                except SystemExit as leak:  # skip this task, not the rest of the sweep
                    print(f"skipping {task['task_id']} at {depth}: {leak}", flush=True)
                    continue
                prompt = longctx.wrap_prompt(prompt, padding)
            before = emit.scrape_metrics(args.base_url)
            started = time.monotonic()
            try:
                generate(args.base_url, prompt, args.max_tokens, args.reasoning_effort, args.timeout)
                failure = None
            except Exception as error:  # a refused request must not look like a slow one
                failure = f"{type(error).__name__}: {error}"
            wall = time.monotonic() - started
            after = emit.scrape_metrics(args.base_url)

            row = {
                "schema_version": emit.SCHEMA_VERSION,
                "timestamp": emit.utc_timestamp(),
                "experiment_id": args.experiment_id,
                "label": args.label,
                "quant": args.quant,
                "served_model_path": probe.get("served_model_path"),
                **env,
                "task_id": task["task_id"],
                "depth_target": depth,
                "depth_padding_tokens": actual,
                "wall_seconds": wall,
                "engine_commit": probe.get("engine_commit"),
                "error": failure,
            }
            row.update(emit.metrics_delta(before, after))
            row.update(acceptance(row))
            emit.append_row(args.out, row)
            print(
                f"{args.label} {task['task_id']} depth={depth} "
                f"tps={row.get('decode_tps')} accept={row.get('draft_accept_rate')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
