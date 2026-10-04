"""Post-hoc egress scan: what the agent ran, read from the Codex rollouts Harbor saved (CPU only).

Task containers have public network (the allowlist is a no-op, CALIBRATION.md), so the agent could fetch the
upstream fix. A hit is a flag for review, not proof: a matched command may only have touched a local path or an
already-installed package. Only call arguments are scanned, never assistant prose or tool output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

PATTERN = re.compile(r"\bpip3? (?:install|download)|\buv (?:pip|add)\b|\bgit (?:fetch|pull|clone|ls-remote)\b|\bcurl\b|\bwget\b"
                     r"|github\.com|githubusercontent|pypi\.org|git log [^|;&]*--all\b")
CALL_TYPES = ("function_call", "custom_tool_call", "local_shell_call")


def call_text(payload):
    """The command text a call carried: JSON arguments, custom-tool input, or a shell argv."""
    if payload.get("type") == "function_call":
        return str(payload.get("arguments") or "")
    if payload.get("type") == "custom_tool_call":
        return str(payload.get("input") or "")
    action = payload.get("action")
    command = action.get("command") if isinstance(action, dict) else None
    return " ".join(map(str, command)) if isinstance(command, list) else str(command or "")


def scan_trial(trial):
    hits, scanned = [], 0
    for path in sorted((Path(trial) / "agent" / "sessions").rglob("rollout-*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            payload = row.get("payload")
            if row.get("type") != "response_item" or not isinstance(payload, dict) or payload.get("type") not in CALL_TYPES:
                continue
            scanned += 1
            text = call_text(payload)
            for match in PATTERN.finditer(text):
                hits.append({"rollout": path.name, "line": number, "pattern": match.group(0),
                             "snippet": text[max(0, match.start() - 60):match.end() + 100]})
    return {"flagged": bool(hits), "hits": hits, "calls_scanned": scanned}


def rescan_campaign(campaign):
    """Rewrite probe.egress_scan in every job file from the saved rollouts; list flagged tasks in egress-flags.json."""
    campaign = Path(campaign)
    flags = []
    for attempt in sorted(campaign.glob("attempts/*")):
        trials = [p.parent for p in attempt.glob("harbor/*/*/result.json")]
        if len(trials) != 1:
            continue
        scan = scan_trial(trials[0])
        for job_file in (attempt / "job.json", *campaign.glob(f"job-agent-{attempt.name.removeprefix('agent-')}.json")):
            if job_file.exists():
                job = json.loads(job_file.read_text())
                job["probe"]["egress_scan"] = scan
                job_file.write_text(json.dumps(job, indent=2) + "\n")
        if scan["flagged"]:
            job = json.loads((attempt / "job.json").read_text())
            flags.append({"task": f'{job["task"]}#r{job["run"]}', "hits": scan["hits"]})
    (campaign / "egress-flags.json").write_text(json.dumps(flags, indent=2) + "\n")
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    flags = rescan_campaign(parser.parse_args().campaign)
    for flag in flags:
        print(f'{flag["task"]}: ' + "; ".join(sorted({h["pattern"] for h in flag["hits"]})))
    print(f"{len(flags)} flagged task(s)")


if __name__ == "__main__":
    main()
