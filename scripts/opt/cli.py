"""python -m scripts.opt <ingest|status|fit|profile|calibrate|report> ..."""
import argparse
import sys
from collections import Counter
from pathlib import Path

from . import fit, goal as goalmod, report as reportmod, workload
from .ingest import run
from .store import Store


def cmd_ingest(a):
    missing = [p for p in a.paths if not Path(p).exists()]
    if missing:
        print("ingest error: path does not exist: " + ", ".join(missing), file=sys.stderr)
        return 2
    rep = run(a.paths, db=a.db, dry_run=a.dry_run)
    for kind, c in sorted(rep["counts"].items()):
        print(f"{kind:16s} units {c['units']:4d}  measurements {c['measurements']:6d}  (invalid {c['invalid']})")
    tot = {k: sum(u[k] for u in rep["units"]) for k in ("added", "updated", "same")}
    print(f"rows: added {tot['added']}, updated {tot['updated']}, unchanged {tot['same']}" + ("  [dry run, nothing written]" if a.dry_run else ""))
    for label, items in (("rejected", rep["rejected"]), ("skipped", rep["skipped"])):
        print(f"{label}: {len(items)}")
        for p, why in items[: a.show]:
            print(f"  {p}\n    {why}")
    print(f"flagged: {len(rep['flagged'])}")
    for kind, p, why in rep["flagged"][: a.show]:
        print(f"  [{kind}] {p}: {why}")
    return 1 if a.strict and rep["rejected"] else 0


def cmd_status(a):
    st = Store(a.db)
    m = st.rows("measurements")
    print(f"db {st.dir}: " + ", ".join(f"{t} {len(st.tables[t])}" for t in st.tables))
    by = Counter((r["source"], r["status"]) for r in m)
    for (src, status), n in sorted(by.items()):
        print(f"  {src:16s} {status:13s} {n}")
    return 0


def cmd_fit(a):
    text, _ = fit.report(Store(a.db))
    print(text)
    return 0


def cmd_profile(a):
    st = Store(a.db)
    for p in workload.build_profiles(st, write=True):
        print(f"{p['id']}: {p['n_campaigns']} source runs, fixed_call_s {p['fixed_call_s']}, task_overhead_s {p.get('task_overhead_s')}")
    st.flush()
    return 0


def cmd_calibrate(a):
    print(workload.report_text(workload.calibrate(Store(a.db))))
    return 0


def cmd_report(a):
    try:
        g = goalmod.load(a.goal)
    except (goalmod.GoalError, OSError) as e:
        print(f"goal error: {e}", file=sys.stderr)
        return 2
    try:
        text, _ = reportmod.render(Store(a.db), g, show_fail=a.all, limit=a.limit)
    except (goalmod.GoalError, ValueError) as e:
        print(f"report error: {e}", file=sys.stderr)
        return 2
    print(text)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m scripts.opt", description=__doc__)
    p.add_argument("--db", help="results DB directory (default results/db, or $OPT_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest", help="parse campaign/weekend/graph-reuse/hq-conc/two-agent/KLD/queue results into the DB (idempotent)")
    s.add_argument("paths", nargs="+")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--strict", action="store_true", help="exit 1 if any source was rejected")
    s.add_argument("--show", type=int, default=25, help="max rejected/skipped/flagged lines printed")
    s.set_defaults(fn=cmd_ingest)
    sub.add_parser("status", help="row counts per source and status").set_defaults(fn=cmd_status)
    sub.add_parser("fit", help="refit VRAM and HQ-pool models; print residuals").set_defaults(fn=cmd_fit)
    sub.add_parser("profile", help="build workload profiles from ingested call tables").set_defaults(fn=cmd_profile)
    sub.add_parser("calibrate", help="wall_est error against recorded campaign walls").set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("report", help="ranked table and Pareto front for a goal.toml")
    s.add_argument("goal")
    s.add_argument("--all", action="store_true", help="also list candidates failing a constraint")
    s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    return a.fn(a)
