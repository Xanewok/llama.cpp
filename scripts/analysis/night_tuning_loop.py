"""Wake periodically to inspect the live Swift tuning state until morning."""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import subprocess
import time as walltime


def morning_deadline(now: datetime, target: time = time(8, 0)) -> datetime:
    deadline = now.replace(hour=target.hour, minute=target.minute,
                           second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    return deadline


def command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return f"error:{type(error).__name__}:{error}"
    return result.stdout.strip()


def units() -> list[dict[str, str]]:
    raw = command(["systemctl", "--user", "list-units", "--all", "--no-legend",
                   "qwen-tuning-*.service"])
    rows = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 4:
            rows.append({"unit": fields[0], "load": fields[1],
                         "active": fields[2], "substate": fields[3]})
    return rows


def gpu_processes() -> list[str]:
    raw = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                   "--format=csv,noheader,nounits"])
    return [line for line in raw.splitlines() if line]


def campaign_states(root: Path) -> list[dict[str, object]]:
    rows = []
    for state_path in sorted(root.glob("campaign-swift-tuning-*/state.json"),
                             key=lambda path: path.stat().st_mtime):
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({"campaign": state_path.parent.name,
                     "status": state.get("status"),
                     "completed": len(state.get("completed", {})),
                     "mtime": state_path.stat().st_mtime})
    return rows


def snapshot(root: Path) -> dict[str, object]:
    current_units = units()
    active = [row for row in current_units
              if row["active"] in {"active", "activating", "reloading"}]
    states = campaign_states(root)
    running = [row for row in states if row["status"] == "running"]
    if active or running:
        reason = "campaign_running"
    else:
        reason = "nothing_to_do"
    return {"time": walltime.time(), "reason": reason,
            "units": current_units, "campaigns": states[-8:],
            "gpu_compute": gpu_processes()}


def run(args: argparse.Namespace) -> int:
    now = datetime.now().astimezone()
    deadline = morning_deadline(now)
    consecutive_nothing = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        now = datetime.now().astimezone()
        if now >= deadline:
            row = {"time": walltime.time(), "reason": "deadline_reached",
                   "deadline": deadline.isoformat()}
            with args.out.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            return 0
        row = snapshot(args.root)
        row["deadline"] = deadline.isoformat()
        with args.out.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if row["reason"] == "nothing_to_do":
            consecutive_nothing += 1
            if consecutive_nothing >= args.nothing_threshold:
                stop = {"time": walltime.time(),
                        "reason": "nothing_to_do_threshold",
                        "consecutive": consecutive_nothing,
                        "deadline": deadline.isoformat()}
                with args.out.open("a") as stream:
                    stream.write(json.dumps(stop) + "\n")
                return 0
        else:
            consecutive_nothing = 0
        if args.once:
            return 0
        remaining = max(0, min(args.interval, (deadline - now).total_seconds()))
        walltime.sleep(remaining)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=28 * 60)
    parser.add_argument("--nothing-threshold", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.interval < 1 or args.nothing_threshold < 1:
        parser.error("interval and nothing-threshold must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
