#!/usr/bin/env python3
"""Campaign sample units; counter-based round metrics; no pooling across trajectories."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

KEYS = ("solved", "wall_s", "output_tokens", "decode_tps", "round_ms", "acceptance", "tokens_per_round", "max_depth")


def campaign(path):
    meta = json.loads((path / "weekend.json").read_text())
    summary = json.loads((path / "SUMMARY.json").read_text())
    jobs = [json.loads(p.read_text()) for p in sorted(path.glob("job-*.json"))]
    passed = [j for j in jobs if j.get("status") == "passed" and j.get("probe")]
    counters = [j["probe"].get("metrics_delta", {}) for j in passed]
    drafts = sum(m.get("drafts", 0) for m in counters)
    generated = sum(m.get("draft_tokens", 0) for m in counters)
    accepted = sum(m.get("draft_accepted_tokens", 0) for m in counters)
    seconds = summary.get("decode_seconds")
    row = {**meta, "path": str(path), "state": summary.get("state"), "jobs_ok": len(passed),
           "solved": summary.get("solved"), "wall_s": summary.get("wall_seconds"),
           "output_tokens": summary.get("output_tokens"),
           "decode_tps": summary["output_tokens"] / seconds if seconds else None,
           "round_ms": 1000 * seconds / drafts if drafts and seconds else None,
           "acceptance": accepted / generated if generated else None,
           "tokens_per_round": 1 + accepted / drafts if drafts else None,
           "max_depth": summary.get("peak_context_tokens"),
           "measured_tasks": sorted(j.get("task", "") for j in passed)}
    row["eligible"] = row["state"] == "complete" and row["jobs_ok"] == 8 and summary.get("tasks") == 8
    return row


def rows(root):
    result = []
    for path in sorted(root.glob("campaign-*")):
        if not (path / "weekend.json").exists():
            continue
        try:
            result.append(campaign(path))
        except (OSError, ValueError, KeyError) as error:
            result.append({"path": str(path), "state": "unavailable", "eligible": False, "error": str(error),
                           **json.loads((path / "weekend.json").read_text())})
    return result


def winner(data, default="S3"):
    groups = defaultdict(list)
    for row in data:
        if row.get("phase") == 1 and row["eligible"] and row.get("decode_tps") is not None:
            groups[row["arm"]].append(row["decode_tps"])
    if set(groups) != {"S0", "S3", "S4", "S5"} or any(len(v) != 3 for v in groups.values()):
        return default, "default; phase 1 does not have three eligible campaigns per arm"
    arm = max(groups, key=lambda a: statistics.mean(groups[a]))
    return arm, "highest mean decode tok/s over three complete passes"


def mean_sem(values):
    if not values:
        return "unknown"
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / len(values)**0.5 if len(values) > 1 else None
    return f"{mean:.6g} +/- {sem:.6g}" if sem is not None else f"{mean:.6g} +/- unknown"


def render(data):
    lines = ["# Weekend campaigns", "", "M = counters, solved, wall, output tokens, max depth. D = ratios, means, SEM, paired deltas. "
             "The sample unit is a campaign. round_ms = 1000 * total decode seconds / verification steps (metrics_delta.drafts); "
             "tokens_per_round = 1 + accepted / verification steps. Decode time includes request boundary overhead. "
             "Failed/retried attempts and fit probes are excluded from timing totals. Incomplete or non-passed jobs remain visible and are excluded from means and winner selection.", "",
             "| Campaign | Phase | Arm | Model | State | Jobs (M) | " + " | ".join(KEYS) + " |",
             "|---|---|---|---|---|---:|" + "---:|" * len(KEYS)]
    for row in data:
        values = ["unknown" if row.get(key) is None else f"{row[key]:.6g}" for key in KEYS]
        lines.append("| " + " | ".join([Path(row["path"]).name, str(row["phase"]), row["arm"], row["model_label"],
                                       row["state"], str(row.get("jobs_ok", 0)), *values]) + " |")
    groups = defaultdict(list)
    for row in data:
        if row["eligible"]:
            groups[(row["phase"], row["arm"], row["model_label"], row["model_sha"])].append(row)
    lines += ["", "Mean +/- SEM by phase, arm, and pinned model (D):", "",
              "| Phase / arm / model / SHA | Campaigns (M) | " + " | ".join(KEYS) + " |",
              "|---|---:|" + "---:|" * len(KEYS)]
    for key, group in sorted(groups.items()):
        values = [mean_sem([r[k] for r in group if r.get(k) is not None]) for k in KEYS]
        lines.append("| " + " | ".join([" / ".join(map(str, key)), str(len(group)), *values]) + " |")
    lines += ["", "Phase-1 within-pass deltas vs S0 (D; arm minus S0):", "",
              "| Pass | Arm | " + " | ".join(KEYS) + " |", "|---:|---|" + "---:|" * len(KEYS)]
    paired = {(r["pair"], r["arm"]): r for r in data if r["eligible"] and r["phase"] == 1}
    for pair in (1, 2, 3):
        base = paired.get((pair, "S0"))
        for arm in ("S3", "S4", "S5"):
            row = paired.get((pair, arm))
            if not (base and row):
                lines.append(f"| {pair} | {arm} | " + " | ".join(["unknown"] * len(KEYS)) + " |")
                continue
            values = [f"{row[k]-base[k]:+.6g}" if row.get(k) is not None and base.get(k) is not None else "unknown" for k in KEYS]
            lines.append(f"| {pair} | {arm} | " + " | ".join(values) + " |")
            if base["measured_tasks"] != row["measured_tasks"]:
                lines.append(f"Task-set mismatch for pass {pair}, {arm}.")
    selected, reason = winner(data)
    lines += ["", f"Phase-1 winner: {selected} ({reason}). "
              "Throughput selection does not establish fidelity; phase 2 reports solved, wall, and token volume separately. "
              "Comparisons use repeated stochastic trajectories, not identical completions."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    data = rows(args.root)
    result = render(data)
    if args.out:
        args.out.write_text(result)
        args.out.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    print(result, end="")


if __name__ == "__main__":
    main()
