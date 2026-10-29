#!/usr/bin/env python3
"""Capture the GPU environment as one JSON object the instruments can record.

Ships over `ssh box python3 - < scripts/gpuenv.py`, so it imports nothing from this
repo and nothing outside the standard library.

Capture once per measured leg, not once per box: the power cap is mutable
(`nvidia-smi -pl`) and is itself an experiment variable, so a per-box capture would
stamp every leg of a wattage sweep with the first cap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

QUERY = "name,driver_version,power.limit"


def parse(smi_csv: str, nvcc_text: str | None, *, allow_mixed: bool = False) -> dict[str, object]:
    """One row per GPU from `nvidia-smi --query-gpu=... --format=csv,noheader`.

    Refuses heterogeneous cards by default. Opt-in mixed captures preserve each
    card's identity and cap; the homogeneous scalar cap must not become an average.
    """

    cards = []
    for line in smi_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            raise ValueError(f"unparsable nvidia-smi row: {line!r}")
        name, driver, power = parts[0], parts[1], parts[2]
        watts = None
        if power not in ("", "[N/A]", "N/A"):
            watts = round(float(power.removesuffix("W").strip()), 1)
        cards.append((name, driver, watts))
    if not cards:
        raise ValueError("nvidia-smi returned no GPUs")

    names = {c[0] for c in cards}
    caps = {c[2] for c in cards}
    mixed = len(names) > 1 or len(caps) > 1
    if mixed and not allow_mixed:
        raise ValueError(
            f"heterogeneous GPUs cannot be described by one row: "
            f"names={sorted(names)} power_limit_w={sorted(map(str, caps))}"
        )

    result = {
        "gpu": " + ".join(c[0] for c in cards) if mixed else cards[0][0],
        "gpu_count": len(cards),
        "gpu_driver": cards[0][1],
        "power_limit_w": None if mixed else cards[0][2],
        # nvidia-smi's "CUDA Version" is the driver's supported API level, not the
        # installed toolkit; only nvcc knows what the engine was compiled against.
        "cuda_toolkit": parse_nvcc(nvcc_text),
    }
    if mixed:
        result["gpu_devices"] = [
            {"gpu": name, "gpu_driver": driver, "power_limit_w": watts}
            for name, driver, watts in cards
        ]
    return result


def parse_nvcc(text: str | None) -> str | None:
    if not text:
        return None
    for token in text.replace(",", " ").split():
        if token.startswith("V") and token[1:2].isdigit():
            return token[1:]
    return None


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def capture(devices: str | None = None, *, allow_mixed: bool = False) -> dict[str, object]:
    # NVML ignores CUDA_VISIBLE_DEVICES. Require an explicit NVML selection so a
    # single-card arm cannot silently inherit every installed GPU's identity.
    if devices is None and "CUDA_VISIBLE_DEVICES" in os.environ:
        raise SystemExit("pass --devices with NVIDIA GPU UUIDs when CUDA_VISIBLE_DEVICES is set")
    command = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader"]
    if devices is not None:
        if not devices.strip():
            raise SystemExit("--devices must select at least one GPU")
        command.append(f"--id={devices}")
    smi = _run(command)
    if smi is None:
        raise SystemExit("nvidia-smi unavailable: cannot identify the GPU")
    return parse(smi, _run(["nvcc", "--version"]), allow_mixed=allow_mixed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", help="nvidia-smi GPU UUIDs or indices, comma-separated")
    parser.add_argument("--allow-mixed", action="store_true",
                        help="record per-device names and caps for heterogeneous arms")
    args = parser.parse_args()
    json.dump(capture(args.devices, allow_mixed=args.allow_mixed), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
