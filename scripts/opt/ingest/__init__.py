"""Source discovery and ingest driver.

A source is classified by its own structure, never by a path convention, so a renamed
or copied directory is still recognised and a look-alike is not.
"""
import json
import os
from pathlib import Path

from ..store import ROOT, Reject, Skip, Store
from . import campaign, graph_reuse, hq_conc, kld, ledger, queue_np, two_agent, weekend

PRUNE = {"attempts", "workspace", "snapshot", "node_modules", "codex-state", ".git", "target", "__pycache__", "tmp",
         "cuda-cache", "texts", "arms", "logs", "remote", "db"}
PARSERS = {"campaign": campaign.parse, "weekend": weekend.parse, "graph-reuse-ab": graph_reuse.parse,
           "hq-conc": hq_conc.parse, "two-agent": two_agent.parse, "kld": kld.parse, "queue-np": queue_np.parse, "ledger": ledger.parse}


def _peek_json(path, limit=1 << 22):
    try:
        if os.path.getsize(path) > limit:
            return None
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def classify_file(p):
    n = p.name
    if n == "PLAN.md":
        return "ledger"
    if n.startswith("results") and n.endswith(".json"):
        d = _peek_json(p)
        return "kld" if isinstance(d, dict) and {"refs", "kld"} <= set(d) else None
    if n == "ab-summary.json":
        return "graph-reuse-ab"
    if n == "summary.json":
        d = _peek_json(p)
        return "weekend" if isinstance(d, list) and d and isinstance(d[0], dict) and "phase" in d[0] and "arm" in d[0] else None
    if n.startswith("results") and n.endswith((".txt", ".json")):
        return "kld"
    if n.endswith(".json") and not n.endswith(".launch.json") and p.with_name(p.stem + ".launch.json").exists():
        d = _peek_json(p)
        return "queue-np" if isinstance(d, dict) and {"arm", "np", "ctx", "results"} <= set(d) else None
    return None


def classify_dir(p):
    if (p / "SUMMARY.json").exists() and (p / "manifest.json").exists():
        d = _peek_json(p / "SUMMARY.json")
        # a two-agent measure/ leg also has job-*.json; the agents table is what identifies it
        if isinstance(d, dict) and "agents" in d:
            return "two-agent"
        return "campaign"
    if (p / "measure" / "SUMMARY.json").exists():
        d = _peek_json(p / "measure" / "SUMMARY.json")
        if isinstance(d, dict) and "agents" in d:
            return "two-agent"
    if (p / "fit.json").exists() and (p / "serve.log").exists():
        return "hq-conc"
    return None


def discover(path, max_depth=7):
    path = Path(path)
    if path.is_file():
        kind = classify_file(path)
        return [(kind, path)] if kind else []
    units = []

    def walk(d, depth):
        kind = classify_dir(d)
        if kind:
            units.append((kind, d))
            return
        if depth >= max_depth:
            return
        for e in sorted(d.iterdir(), key=lambda x: x.name):
            if e.is_dir() and not e.is_symlink() and e.name not in PRUNE:
                walk(e, depth + 1)
            elif e.is_file():
                k = classify_file(e)
                if k:
                    units.append((k, e))

    walk(path, 0)
    return units


def _rel(v):
    if isinstance(v, str) and v.startswith("/"):
        try:
            return str(Path(v).resolve().relative_to(ROOT))
        except ValueError:
            return v
    return v


def _normalise_paths(row):
    holder = row.get("provenance", row)
    for k in ("out_dir", "summary_path"):
        if k in holder:
            holder[k] = _rel(holder[k])


def run(paths, db=None, dry_run=False):
    """Returns a report dict; raises nothing for per-unit rejects (they are listed)."""
    store = Store(db)
    rep = {"units": [], "rejected": [], "skipped": [], "flagged": [], "counts": {}}
    for root in map(lambda x: str(Path(x).resolve()), paths):
        if not Path(root).exists():
            rep["rejected"].append((str(root), "path does not exist"))
            continue
        units = discover(root)
        if not units:
            rep["rejected"].append((str(root), "no recognisable source (campaign/weekend/graph-reuse/hq-conc/two-agent/kld/queue-np)"))
        for kind, p in units:
            try:
                b = PARSERS[kind](p)
            except Skip as e:
                rep["skipped"].append((str(p), f"[{kind}] {e}"))
                continue
            except Reject as e:
                rep["rejected"].append((str(p), f"[{kind}] {e}"))
                continue
            res = {"added": 0, "updated": 0, "same": 0}
            for table, rows in (("configs", b.configs.values()), ("workloads", b.workloads.values()),
                                ("measurements", b.measurements)):
                for r in rows:
                    r.setdefault("id", r.get("config_id"))
                    if table != "configs":
                        _normalise_paths(r)
                    got = store.upsert(table, r)
                    if table == "measurements":
                        res[got] += 1
            rep["units"].append({"kind": kind, "path": str(p), "measurements": len(b.measurements), **res})
            rep["flagged"] += [(kind,) + f for f in b.flagged]
            c = rep["counts"].setdefault(kind, {"units": 0, "measurements": 0, "invalid": 0})
            c["units"] += 1
            c["measurements"] += len(b.measurements)
            c["invalid"] += sum(m["status"] == "invalid" for m in b.measurements)
    if not dry_run:
        store.flush()
    return rep
