"""JSONL tables, canonical TOML and stable ids.

Contract: every row has a deterministic `id`; upsert replaces by id, so re-ingesting the
same source rewrites identical bytes. Files are written sorted by id via rename (no
partial files).
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ("measurements", "configs", "workloads", "levers")
NATURAL_KEY = {"measurements": "id", "configs": "config_id", "workloads": "id", "levers": "lever_id"}
STATUSES = ("measured", "derived", "ledger-prior", "invalid")


class Reject(Exception):
    """A source file is corrupt/partial and must not enter the DB."""


class Skip(Reject):
    """Recognised file that deliberately yields nothing (e.g. a results table with no stated reference)."""


def db_dir(override=None):
    return Path(override or os.environ.get("OPT_DB") or ROOT / "results" / "db")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(*parts, n=16):
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:n]


def _toml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    raise TypeError(f"not canonical-TOML encodable: {type(v)}")


def canonical_toml(d, _prefix=()):
    """Sorted keys, scalars before tables, no comments: byte-identical for equal dicts."""
    scalars = {k: v for k, v in d.items() if not isinstance(v, dict) and v is not None}
    tables = {k: v for k, v in d.items() if isinstance(v, dict) and v}
    out = []
    if _prefix and scalars:
        out.append("[" + ".".join(json.dumps(p) if not p.isidentifier() else p for p in _prefix) + "]")
    for k in sorted(scalars):
        key = k if k.replace("_", "").replace("-", "").isalnum() else json.dumps(k)
        out.append(f"{key} = {_toml_scalar(scalars[k])}")
    text = "\n".join(out) + ("\n" if out else "")
    for k in sorted(tables):
        text += canonical_toml(tables[k], _prefix + (k,))
    return text


def config_id_of(canon):
    return hashlib.sha256(canonical_toml(canon).encode()).hexdigest()[:16]


def _paths(row):
    h = row.get("provenance", row)
    return [h.get("summary_path") or h.get("out_dir")] + list(h.get("also_at", []))


PATHISH = ("out_dir", "summary_path", "summary_sha256", "also_at", "item_id", "pair_group", "attempt", "campaign")


def _content(row):
    r = json.loads(json.dumps(row))
    holder = r.get("provenance", r)
    for k in PATHISH:
        holder.pop(k, None)
        r.pop(k, None)
    return r


def _canonical_first(paths):
    return sorted({x for x in paths if x}, key=lambda x: (x.startswith("/"), x))


def _merge_paths(old, new):
    """Same content, possibly another copy: keep the canonical copy's path-derived fields
    (repo-relative first, then lexicographic) and list the rest in also_at."""
    allp = _canonical_first(_paths(old) + _paths(new))
    base = json.loads(json.dumps(new if _paths(new)[0] == (allp[0] if allp else None) else old))
    holder = base.get("provenance", base)
    if len(allp) > 1:
        holder["also_at"] = allp[1:]
    else:
        holder.pop("also_at", None)
    return base


class Store:
    def __init__(self, directory=None):
        self.dir = db_dir(directory)
        self.tables = {t: self._load(t) for t in TABLES}
        self.dirty = set()

    def _path(self, t):
        return self.dir / f"{t}.jsonl"

    def _load(self, t):
        rows = {}
        p = self._path(t)
        if p.exists():
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise Reject(f"{p}:{i}: corrupt DB row ({e})")
                    rid = r.get("id") or r.get(NATURAL_KEY[t])
                    if not rid and t == "levers":
                        rid = "~" + stable_id(line)  # foreign table: read-only here, never rewritten
                    if not rid:
                        raise Reject(f"{p}:{i}: row has neither id nor {NATURAL_KEY[t]}")
                    rows[rid] = r
        return rows

    def upsert(self, table, row):
        """Same content seen at another path (archive copy) merges its path into `also_at`."""
        old = self.tables[table].get(row["id"])
        if old is not None and _content(old) == _content(row):
            row = _merge_paths(old, row)
        if old != row:
            self.tables[table][row["id"]] = row
            self.dirty.add(table)
            return "updated" if old else "added"
        return "same"

    def flush(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        for t in sorted(self.dirty):
            p = self._path(t)
            tmp = p.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as f:
                for rid in sorted(self.tables[t]):
                    f.write(json.dumps(self.tables[t][rid], sort_keys=True, separators=(",", ":")) + "\n")
            os.replace(tmp, p)
        self.dirty.clear()

    def rows(self, table):
        return list(self.tables[table].values())
