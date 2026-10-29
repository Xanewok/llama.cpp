"""Periodic evidence-only summary of the fixed local overnight campaigns."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import time

from scripts.instruments.hillclimb import atomic_json, read_sealed


def read_jobs(campaign):
    if not campaign.exists():
        return []
    return [(path.stem[4:], read_sealed(path)) for path in sorted(campaign.glob("job-*.json"))]


def coding_rows(agentic, corrected_bartowski=None):
    original = read_jobs(agentic)
    if corrected_bartowski is None:
        return original
    expected = {name: row['task'] for name, row in original if row.get('model') == 'bartowski-iq4-xs'}
    manifest = read_sealed(corrected_bartowski / 'manifest.json')
    declared = {job['id']: job['task'] for job in manifest['queue']}
    if (not expected or declared != expected or len(declared) != len(manifest['queue']) or
            any(job['model'] != 'bartowski-iq4-xs' for job in manifest['queue'])):
        raise ValueError('corrected campaign must cover exactly the original Bartowski tasks')
    corrected = read_jobs(corrected_bartowski)
    for name, row in corrected:
        if name not in expected or row.get('model') != 'bartowski-iq4-xs' or row.get('task') != expected[name]:
            raise ValueError('corrected result does not match its declared Bartowski job')
    # Replace the entire incompatible arm, never fill pending jobs from the old failures.
    return [(name, row) for name, row in original if row.get('model') != 'bartowski-iq4-xs'] + corrected


def summarize(root, corrected_bartowski=None):
    throughput = root / "campaign-remaining-quants-20260913-r3"
    agentic = root / "campaign-agentic-20260913"
    rows = read_jobs(root / "campaign-20260912-b") + read_jobs(throughput)
    repeated, slots = defaultdict(list), []
    for name, row in rows:
        if name.startswith("repeat-"):
            repeated[row.get("model", "unknown")].append(row)
        if name.startswith("slots-"):
            slots.append(row)
    quant_stats = []
    for model, records in sorted(repeated.items()):
        passed = [r for r in records if r["status"] == "passed"]
        speeds = [r["decode_tps"] for r in passed]
        quant_stats.append({"model": model, "processed_repeats": len(records), "passed_repeats": len(passed),
                           "median_tg": statistics.median(speeds) if speeds else None,
                           "median_pp": statistics.median(r["prefill_tps"] for r in passed) if passed else None,
                           "minimum_tg": min(speeds) if speeds else None, "maximum_tg": max(speeds) if speeds else None,
                           "qualified_repeat_speed": len(passed) >= 12 and statistics.median(speeds) >= 35})
    tasks = defaultdict(list)
    for _, row in coding_rows(agentic, corrected_bartowski):
        tasks[row.get("model", "unknown")].append(row)
    task_stats, paired = [], []
    baseline = {r["task"]: r for r in tasks.get("iq3-control", [])
                if r.get("probe", {}).get("task_success")}
    for model, records in sorted(tasks.items()):
        measured = [r for r in records if r["status"] == "passed" and r.get("probe", {}).get("complete")]
        probes = [r["probe"] for r in measured]
        wall = sum(p["verified_wall_seconds"] for p in probes)
        prefill = sum(p["prefill_seconds"] for p in probes)
        decode = sum(p["decode_seconds"] for p in probes)
        task_stats.append({"model": model, "processed": len(records), "measured_tasks": len(probes),
                           "solved": sum(p["task_success"] for p in probes),
                           "timed_out": sum(p["timed_out"] for p in probes),
                           "wall_seconds": wall, "prefill_seconds": prefill, "decode_seconds": decode,
                           "other_seconds": max(0, wall - prefill - decode),
                           "prefill_fraction": prefill / wall if wall else None,
                           "decode_fraction": decode / wall if wall else None,
                           "output_tokens": sum(p["metrics_delta"].get("decode_tokens", 0) for p in probes),
                           "sampled_board_wh": sum(r.get("sampled_board_energy_wh", 0) for r in measured)})
        comparisons = [(baseline[r["task"]]["probe"]["verified_wall_seconds"], r["probe"]["verified_wall_seconds"])
                       for r in measured if r["probe"]["task_success"] and r["task"] in baseline and model != "iq3-control"]
        if comparisons:
            paired.append({"model": model, "both_solved_pairs": len(comparisons),
                           "candidate_over_baseline_geomean": math.exp(statistics.mean(math.log(b / a) for a, b in comparisons)),
                           "candidate_over_baseline_total": sum(b for a, b in comparisons) / sum(a for a, b in comparisons)})
    states = {}
    campaigns = [throughput, agentic] + ([corrected_bartowski] if corrected_bartowski else [])
    for campaign in campaigns:
        for filename in ("state.json", "stage-status.json"):
            if corrected_bartowski and filename == 'stage-status.json':
                continue
            path = campaign / filename
            if path.exists():
                state = json.loads(path.read_text())
                states[campaign.name + "/" + filename] = {
                    "status": state.get("status", state.get("phase")), "reason": state.get("reason"),
                    "completed": len(state.get("completed", {})),
                    "active": state.get("active", {}).get("id") if state.get("active") else None}
    return {"updated_epoch": time.time(), "states": states, "near_100k_repeats": quant_stats,
            "corrected_bartowski": str(corrected_bartowski) if corrected_bartowski else None,
            "task_details": {model: {r['task']: {
                "status": r['status'], "reason": r.get('reason'),
                "solved": r.get('probe', {}).get('task_success', False),
                "wall_seconds": r.get('probe', {}).get('verified_wall_seconds')}
                for r in records} for model, records in tasks.items()},
            "concurrency_jobs_processed": len(slots),
            "concurrency_jobs_passed": sum(r["status"] == "passed" for r in slots),
            "real_task_totals": task_stats, "paired_both_solved_only": paired}


def render(summary):
    title = "# Coding pilot comparison — corrected Bartowski" if summary.get('corrected_bartowski') else "# Overnight evidence report"
    lines = [title, "", "Updated: " + time.strftime("%Y-%m-%d %H:%M:%S %Z"), "",
             "Live results are provisional until queues finish. No automatic production-profile promotion.", "",
             "## Queue state", ""]
    for name, state in summary["states"].items():
        lines.append(f"- `{name}`: {state['status']}; {state['completed']} recorded jobs; active `{state['active']}`; reason `{state['reason']}`.")
    if summary.get('corrected_bartowski'):
        lines += ["", "Bartowski coding source: `" + summary['corrected_bartowski'] + "`.",
                  "This replaces the original incompatible-template arm for comparison; original failures remain unchanged on disk.",
                  "The two successful template-validation solves are reused, not repeated. The other arms are the overnight observations.",
                  "Daytime desktop activity and chronological order differ; this is a small paired pilot, not a randomized replicated trial."]
    lines += ["", "## Repeated near-100k throughput", "",
              "Each request uses about 98k input and up to 2,048 output; these are not graded solves.",
              "Desktop and CPU activity varied; compare repeated medians, not a single fastest sample.", "",
              "| Quant | Valid / processed repeats | Median PP | Median TG | TG range | >=35 TG with 12 valid repeats |",
              "|---|---:|---:|---:|---:|---|"]
    for row in summary["near_100k_repeats"]:
        fmt = lambda key: "" if row[key] is None else f"{row[key]:.2f}"
        lines.append(f"| {row['model']} | {row['passed_repeats']} / {row['processed_repeats']} | {fmt('median_pp')} | {fmt('median_tg')} | {fmt('minimum_tg')}–{fmt('maximum_tg')} | {row['qualified_repeat_speed']} |")
    lines += ["", f"Concurrent 100k-per-slot configurations: {summary['concurrency_jobs_passed']} passed out of {summary['concurrency_jobs_processed']} processed.",
              "Failure applies to the tested split/batch/cache settings, not every possible concurrency configuration.", "",
              "## Real coding-task pilot", "",
              "Natural tool-loop history, fresh isolated workspaces, trusted external tests, MTP on, no throughput output cap.",
              "Totals include failed/timed-out measured tasks; fit skips and instrument failures are not coding failures.", "",
              "Totals with different task counts are not directly comparable; use the matched-task ratios below.",
              "The single-IQ3 control has 65,536 configured context and paired candidates have 102,400; this compares complete setups, not the GPU alone.", "",
              "| Quant | Solved / measured / processed | Timeouts | Wall s | Prefill s (%) | Decode s (%) | Other s | Output tokens | GPU-board Wh |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["real_task_totals"]:
        pp = row["prefill_fraction"] or 0
        tg = row["decode_fraction"] or 0
        lines.append(f"| {row['model']} | {row['solved']} / {row['measured_tasks']} / {row['processed']} | {row['timed_out']} | {row['wall_seconds']:.1f} | {row['prefill_seconds']:.1f} ({pp:.1%}) | {row['decode_seconds']:.1f} ({tg:.1%}) | {row['other_seconds']:.1f} | {row['output_tokens']} | {row['sampled_board_wh']:.2f} |")
    if not summary["real_task_totals"]:
        lines += ["", "No real coding-task result has been recorded yet."]
    lines += ["", "Board energy includes model startup and both installed boards, including idle 2060 power in the single-card control; it is not wall-plug energy.",
              "Task wall time includes trusted grading but excludes server startup. Inspect failed/retried attempt logs separately for recovery cost.", "",
              "## Paired time to verified solution", "",
              "Only tasks solved by both arms enter these descriptive ratios (candidate / single-IQ3 baseline; lower is faster).",
              "This is conditional on both solving, not an overall success-adjusted ranking. Small or incomplete pilots do not establish superiority."]
    for row in summary["paired_both_solved_only"]:
        lines.append(f"- {row['model']}: {row['both_solved_pairs']} paired solves; geometric ratio {row['candidate_over_baseline_geomean']:.3f}; ratio of totals {row['candidate_over_baseline_total']:.3f}.")
    if summary.get('corrected_bartowski'):
        columns = [('iq3-control', 'Single IQ3'), ('bartowski-iq4-xs', 'Bartowski IQ4'),
                   ('unsloth-q4-m', 'Q4_K_M'), ('unsloth-iq4-xs', 'Unsloth IQ4'),
                   ('jpetrina-iq4-pure', 'jpetrina IQ4')]
        details = summary['task_details']
        lines += ['', '## Per-task verified wall seconds', '',
                  'Only solved tasks display a time. Pending/failed jobs are not zero-duration solves.', '',
                  '| Task | ' + ' | '.join(label for _, label in columns) + ' |',
                  '|---|' + '---:|' * len(columns)]
        for task in sorted(details.get('iq3-control', {})):
            values = []
            for model, _ in columns:
                row = details.get(model, {}).get(task)
                values.append('pending' if row is None else
                              f"{row['wall_seconds']:.2f}" if row['status'] == 'passed' and row['solved'] else
                              row.get('reason') or ('failed tests' if row['status'] == 'passed' else row['status']))
            lines.append('| ' + ' | '.join([task, *values]) + ' |')
    lines += ["", "## Boundaries", "",
              "The single-card fallback is unchanged. Browser closure, driver/P2P/BIOS changes and lower target KV precision are not deployment assumptions.",
              "The old .meta-exposed 78-task runs are not clean quality controls. This eight-task pilot is not a replacement for the later hardened full 78-task evaluation.",
              "See AUTONOMOUS.md and AVENUES.md for outstanding MTP-confidence/draft-KV, NCCL, placement, context-ceiling and soak experiments.", ""]
    return "\n".join(lines)


def write_report(root, corrected_bartowski=None, stem='OVERNIGHT'):
    summary = summarize(root, corrected_bartowski)
    atomic_json(root / (stem + '-SUMMARY.json'), summary)
    temporary = root / (stem + '-REPORT.md.tmp')
    temporary.write_text(render(summary))
    temporary.replace(root / (stem + '-REPORT.md'))


def campaign_terminal(campaign):
    # An error may still be recovered by the bounded service restart policy.
    return json.loads((campaign / 'state.json').read_text())['status'] in ('complete', 'stopped')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument('--corrected-bartowski', type=Path)
    parser.add_argument('--output-stem')
    parser.add_argument('--until-complete', action='store_true')
    args = parser.parse_args()
    stem = args.output_stem or ('PILOT-COMPARISON' if args.corrected_bartowski else 'OVERNIGHT')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', stem) or (args.corrected_bartowski and stem == 'OVERNIGHT'):
        parser.error('use a separate safe filename stem for corrected results')
    if args.until_complete and (not args.corrected_bartowski or not args.deadline_epoch):
        parser.error('--until-complete requires a corrected campaign and a fixed deadline')
    next_report = 0
    while True:
        now = time.time()
        terminal = args.until_complete and campaign_terminal(args.corrected_bartowski)
        if now >= next_report or terminal:
            write_report(args.root, args.corrected_bartowski, stem)
            next_report = now + 1800
        if args.deadline_epoch is None or terminal:
            break
        if now >= args.deadline_epoch + 30:
            write_report(args.root, args.corrected_bartowski, stem)
            break
        time.sleep(min(30, max(0, args.deadline_epoch + 30 - now)))


if __name__ == "__main__":
    main()
