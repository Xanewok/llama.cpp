"""Shared ingest plumbing: Batch accumulator and measurement row builder."""
import json
import re
from pathlib import Path

from ..store import Reject, STATUSES, sha256_file, stable_id

PARSER_VERSION = 1


class Batch:
    def __init__(self):
        self.measurements, self.configs, self.workloads, self.flagged = [], {}, {}, []

    def add_config(self, rec):
        self.configs[rec["id"]] = rec
        return rec["id"]

    def add_workload(self, rec):
        self.workloads[rec["id"]] = rec

    def flag(self, path, msg):
        self.flagged.append((str(path), msg))

    def extend(self, other):
        self.measurements += other.measurements
        self.configs.update(other.configs)
        self.workloads.update(other.workloads)
        self.flagged += other.flagged


def measurement(*, source, metric, value, unit, config_id, workload_id, prov, dims=None, n=1, spread=None,
                status="measured", reason=None, item_id=None, attempt=None, ts=None, host=None,
                pair_group=None, key=(), ident=None):
    """`key` disambiguates rows that share (source, config, workload, metric, dims) inside one source."""
    assert status in STATUSES, status
    dims = dims or {}
    if isinstance(value, float) and value != value:
        raise Reject(f"NaN value for {metric}")
    row = {
        "id": stable_id(source, ident or prov.get("summary_sha256") or prov.get("out_dir"), config_id, workload_id, metric, json.dumps(dims, sort_keys=True), *key),
        "ts": ts, "source": source, "item_id": item_id, "attempt": attempt, "config_id": config_id,
        "workload_id": workload_id, "host": host or {}, "metric": metric, "dims": dims,
        "value": value, "unit": unit, "n": n, "spread": spread, "pair_group": pair_group,
        "status": status, "provenance": {**prov, "parser": source, "parser_version": PARSER_VERSION},
    }
    if reason:
        row["reason"] = reason
    return row


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Reject(f"{path}: unreadable/corrupt JSON ({e})")


def need(d, keys, where):
    miss = [k for k in keys if not isinstance(d, dict) or k not in d]
    if miss:
        raise Reject(f"{where}: missing required keys {miss}")


def verify_sha_sidecar(path):
    """job-*.json.sha256 / manifest.json.sha256 hold the hex digest of the sibling file."""
    side = Path(str(path) + ".sha256")
    if not side.exists():
        return None
    want = side.read_text().strip().strip('"').split()[0]
    got = sha256_file(path)
    if want != got:
        raise Reject(f"{path}: sha256 sidecar mismatch (recorded {want[:12]}, actual {got[:12]})")
    return got


def pair_name(name):
    ph, pr = re.search(r"phase(\d+)", name), re.search(r"pair(\d+)", name)
    if ph and pr:
        return f"phase{ph.group(1)}-pair{pr.group(1)}"
    m = re.search(r"(pair|rep|r)(\d+)", name)
    return m.group(0) if m else None
