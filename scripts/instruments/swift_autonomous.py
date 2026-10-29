"""Bounded unattended continuation of the existing Swift tuning screens."""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import statistics
import subprocess
import time as walltime

from scripts.instruments import swift_longtail


SERVE = Path("/home/xanewok/.codex/skills/local-qwen-server/bin/serve.sh")
WRAPPER = Path("/home/xanewok/repos/qwen38-eval/scripts/instruments/run_swift_tuning_supervised.sh")
STAGES = ("batch", "mtp", "probability", "minimum", "draft-cache", "swarm-allocation")
PROFILE_KEYS = frozenset({
    "context", "slots", "mode", "split", "ubatch", "maximum", "chat_template",
    "batch", "minimum", "p_min", "draft_k", "draft_v",
})


def deadline(now: datetime, target: time = time(8, 0)) -> datetime:
    value = now.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    return value if value > now else value + timedelta(days=1)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def run_command(args: list[str], *, check=False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=30, check=check)


def stop_interactive(log) -> None:
    """Stop only our interactive unit; an already absent unit is a safe no-op."""
    stopped = run_command([str(SERVE), "stop"])
    if stopped.returncode == 0:
        return
    active = run_command(["systemctl", "--user", "is-active", "qwen-local-server.service"])
    if active.stdout.strip() in {"active", "activating", "deactivating"}:
        raise RuntimeError(f"interactive stop failed: {stopped.stderr.strip()}")
    log({"reason": "interactive_already_stopped", "detail": stopped.stderr.strip()})


def campaign_status(path: Path) -> dict:
    state = read_json(path / "state.json", {}) or {}
    return {"status": state.get("status"),
            "completed": len(state.get("completed", {})),
            "total": len((read_json(path / "manifest.json", {}) or {}).get("queue", [])),
            "reason": state.get("reason")}


def passing_results(path: Path) -> list[dict]:
    rows = []
    for result_path in path.glob("attempts/*/result.json"):
        result = read_json(result_path)
        if result and result.get("status") == "passed" and result.get("target", 0) > 0:
            rows.append(result)
    return rows


def profile_config(config: dict) -> dict:
    return {key: config[key] for key in PROFILE_KEYS if key in config}


def result_score(row: dict):
    if row.get("decode_tps") is not None:
        return row["decode_tps"]
    return ((row.get("probe") or {}).get("metrics_delta") or {}).get("decode_tps")


def same_profile(left: dict, right: dict) -> bool:
    return profile_config(left) == profile_config(right)


def config_key(config: dict, stage: str):
    if stage == "batch":
        return config.get("batch", 2048)
    if stage == "mtp":
        return config.get("maximum", 3)
    if stage == "probability":
        return config.get("p_min", 0)
    if stage == "minimum":
        return config.get("minimum", 0)
    if stage == "draft-cache":
        return (config.get("draft_k", "f16"), config.get("draft_v", "f16"))
    return None


def select_config(path: Path, base: dict, stage: str) -> tuple[dict, dict]:
    rows = passing_results(path)
    prefixes = {
        "batch": "batch-batch-",
        "mtp": "mtp-maximum-",
        "probability": "probability-p_min-",
        "minimum": "minimum-minimum-",
        "draft-cache": "draft-cache-draft-q8-experimental-",
    }
    prefix = prefixes.get(stage)
    candidate_rows = rows
    if prefix:
        candidate_rows = [row for row in rows if str(row.get("job", "")).startswith(prefix)]
    grouped: dict[object, list[dict]] = {}
    for row in candidate_rows:
        grouped.setdefault(config_key(row.get("config", {}), stage), []).append(row)
    candidates = []
    for key, values in grouped.items():
        coverage = {(row.get("task"), row.get("target")) for row in values}
        if len(coverage) < 4:
            continue
        scores = [result_score(row) for row in values if result_score(row) is not None]
        if len(scores) < 4:
            continue
        representative = profile_config(values[0]["config"])
        candidates.append((statistics.mean(scores), key, representative, len(values)))
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    base_rows = [row for row in rows if same_profile(row.get("config", {}), base)]
    matched_prefix = f"{stage}-matched-control-"
    matched_rows = [row for row in base_rows if str(row.get("job", "")).startswith(matched_prefix)]
    if matched_rows:
        base_rows = matched_rows
    base_coverage = {(row.get("task"), row.get("target")) for row in base_rows}
    base_scores = [result_score(row) for row in base_rows if result_score(row) is not None]
    base_mean = statistics.mean(base_scores) if len(base_coverage) >= 4 and len(base_scores) >= 4 else None
    eligible = [candidate for candidate in candidates
                if base_mean is not None and candidate[0] >= base_mean * 1.05]
    if eligible:
        score, key, selected, count = eligible[0]
        return selected, {"stage": stage, "selected": key, "mean_decode_tps": score,
                          "coverage_rows": count, "candidates": len(candidates),
                          "baseline_mean_decode_tps": base_mean,
                          "minimum_gain_fraction": 0.05,
                          "decision": "highest mean decode among fully covered arms beating base by at least 5%"}
    decision = "retain base; no fully covered passing candidate"
    if candidates and base_mean is None:
        decision = "retain base; base coverage is incomplete"
    elif candidates:
        decision = "retain base; no candidate beats base by at least 5%"
    return profile_config(base), {"stage": stage, "selected": config_key(base, stage),
                                  "mean_decode_tps": None, "coverage_rows": 0,
                                  "candidates": len(candidates),
                                  "baseline_mean_decode_tps": base_mean,
                                  "minimum_gain_fraction": 0.05, "decision": decision}


def prepare_and_launch(root: Path, selected: dict, stage: str, now_tag: str,
                       log, budget_hours: float) -> tuple[Path, str]:
    queue = root / f"swift-autonomous-{stage}-{now_tag}-queue.json"
    swift_longtail.tuning_main([stage, "--selected-config",
                                str(root / "autonomous-selected.json"), "--out", str(queue)])
    campaign = root / f"campaign-swift-autonomous-{stage}-{now_tag}"
    run_command(["python3", "-m", "scripts.instruments.hillclimb", "prepare",
                 "--swift-tuning", "--queue-json", str(queue), "--campaign", str(campaign)], check=True)
    stop_interactive(log)
    unit = f"qwen-tuning-autonomous-{stage}-{now_tag}"
    launched = run_command(["systemd-run", "--user", "--no-block", f"--unit={unit}",
                            "--collect", "--service-type=oneshot",
                            "--working-directory=/home/xanewok/repos/qwen38-eval",
                            str(WRAPPER), str(campaign), str(budget_hours)])
    if launched.returncode:
        raise RuntimeError(f"systemd-run failed: {launched.stdout.strip()} {launched.stderr.strip()}")
    log({"reason": "launched", "stage": stage, "unit": unit, "campaign": str(campaign),
         "queue": str(queue)})
    return campaign, unit


def log_event(out: Path, row: dict):
    row = {"time": walltime.time(), **row}
    out.with_name(out.stem + "-latest.json").write_text(json.dumps(row, indent=2) + "\n")
    with out.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--current-campaign", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--nothing-threshold", type=int, default=3)
    parser.add_argument("--budget-hours", type=float, default=8)
    args = parser.parse_args(argv)
    if args.interval < 1 or args.nothing_threshold < 1:
        parser.error("interval and nothing-threshold must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.root / "autonomous-dispatch-state.json"
    state = read_json(state_path, {}) or {"stage_index": 0, "campaign": str(args.current_campaign),
                                          "base_config": str(args.base_config), "nothing": 0}
    current_file = args.root / "autonomous-current-campaign.txt"
    current_file.write_text(str(state["campaign"]) + "\n")
    stop_at = deadline(datetime.now().astimezone())

    def log(row):
        log_event(args.out, row)

    while datetime.now().astimezone() < stop_at:
        campaign = Path(state["campaign"])
        status = campaign_status(campaign)
        log({"reason": "check", "stage": STAGES[state["stage_index"]],
             "campaign": str(campaign), "status": status})
        if status["status"] in {"prepared", "running"}:
            state["nothing"] = 0
        elif status["status"] == "complete":
            stage = STAGES[state["stage_index"]]
            if stage == "swarm-allocation":
                log({"reason": "no_auto_promotion", "stage": stage,
                     "detail": "swarm allocation requires separate concurrency qualification"})
                state["nothing"] += 1
            else:
                base = profile_config(read_json(Path(state["base_config"]), {}) or {})
                selected, evidence = select_config(campaign, base, stage)
                selected = profile_config(selected)
                selected_path = args.root / "autonomous-selected.json"
                selected_path.write_text(json.dumps(selected, indent=2) + "\n")
                log({"reason": "selected", "stage": stage, "evidence": evidence,
                     "config": selected})
                next_index = state["stage_index"] + 1
                if next_index >= len(STAGES):
                    state["nothing"] += 1
                else:
                    next_stage = STAGES[next_index]
                    try:
                        next_campaign, _ = prepare_and_launch(
                            args.root, selected, next_stage,
                            datetime.now().strftime("%Y%m%d-%H%M%S"), log, args.budget_hours)
                    except Exception as error:
                        log({"reason": "dispatch_error", "stage": next_stage,
                             "error": f"{type(error).__name__}: {error}"})
                        state["nothing"] += 1
                    else:
                        state.update(stage_index=next_index, campaign=str(next_campaign),
                                     base_config=str(selected_path), nothing=0)
                        current_file.write_text(str(next_campaign) + "\n")
        else:
            state["nothing"] += 1
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        if state["nothing"] >= args.nothing_threshold:
            log({"reason": "nothing_to_do_threshold", "consecutive": state["nothing"]})
            return 0
        remaining = max(0, min(args.interval,
                               (stop_at - datetime.now().astimezone()).total_seconds()))
        walltime.sleep(remaining)
    log({"reason": "deadline_reached"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
