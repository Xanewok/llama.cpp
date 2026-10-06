# Recorded artifacts

These are copies of real GPU-run artifacts, not generated CPU fixture output.

- `perf-real.sql`: `~/repos/qwen38-eval-worktrees/wt/mmvq-np2/codex/fixtures/perf-real.sql`; its original source is the generic wide-MMVQ baseline GPU probe, build `94cceb8b8`, dated 2026-10-04 in the SQL rows.
- `stock-real.sql`: `~/repos/qwen38-eval/results/2026-10-04-mmvq-baseline/a1/raw/stock-r1.sql`.
- `memcheck-real.txt`: `~/repos/qwen38-eval-worktrees/wt/mmvq-np2/codex/fixtures/memcheck-real.txt`, recorded compute-sanitizer output on the RTX 5060 Ti.
- `journal-real.txt`: `~/repos/qwen38-eval/orchestration/gpuq/fixtures/xid-lines.txt`.
- `duo-depth-real.json`: `~/repos/qwen38-eval/results/2026-10-04-duo-np2-depth/full-a3/0-up/depth-8192.json`.

The shared depth parser also replays `~/repos/qwen38-eval/orchestration/gpuq/fixtures/` directly. New small-J path signatures and weight-rotation evidence are live canary gates; no recorded execution of the new kernel exists yet.
