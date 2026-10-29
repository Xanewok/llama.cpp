"""Weekend summary.json: list of campaign rows (phase/arm/pair) -> campaign parser + labels.

When a row's campaign directory exists it is parsed by the campaign parser (full launch
record, per-task data) and the summary adds the round-ms/acceptance columns. Without the
directory the config is rebuilt from the row's `extra` flags and marked as such.
"""
from pathlib import Path

from .. import configs
from ..store import Reject
from . import campaign
from .base import Batch, load_json, measurement, need, verify_sha_sidecar  # noqa: F401
from ..store import sha256_file

REQUIRED = ["phase", "arm", "pair", "model_sha", "commit", "k", "path", "state", "wall_s", "eligible"]


def _find_dir(summary_dir, row):
    for cand in (Path(row["path"]), summary_dir / Path(row["path"]).name):
        if (cand / "SUMMARY.json").exists():
            return cand
    return None


def parse(path):
    sp = Path(path)
    rows = load_json(sp)
    if not isinstance(rows, list) or not rows:
        raise Reject(f"{sp}: weekend summary must be a non-empty JSON list")
    out = Batch()
    ssha = sha256_file(sp)
    for row in rows:
        try:
            need(row, REQUIRED, f"{sp} row phase={row.get('phase') if isinstance(row, dict) else '?'}")
            cdir = _find_dir(sp.parent, row)
            tag = f"phase{row['phase']}-{row['arm']}-pair{row['pair']}-{row['model_label']}"
            if cdir is not None:
                b = campaign.parse(cdir)
                cid = next(iter(b.configs))
            else:
                b = Batch()
                cmd = ["llama-server", "-m", row["model_path"], "--spec-draft-n-max", str(row["k"])]
                rec = configs.from_command(cmd, None, build_sha=row["commit"], model_sha256=row["model_sha"],
                                           extra_flags=row.get("extra"), stack="weekend-summary-only")
                cid = b.add_config(rec)
                b.flag(sp, f"{tag}: campaign dir missing; config rebuilt from summary flags (no launch record)")
            prov = {"out_dir": str(cdir or row["path"]), "summary_path": str(sp), "summary_sha256": ssha,
                    "build_sha": row["commit"], "model_sha256": row["model_sha"]}
            wl = campaign.workload_id(row.get("measured_tasks") or [])
            tag = cdir.name if cdir is not None else tag
            common = dict(source="weekend", config_id=cid, workload_id=wl, prov=prov, item_id=tag, key=(row["phase"], row["arm"], row["pair"], row["model_label"]),
                          pair_group=f"phase{row['phase']}-pair{row['pair']}")
            if cdir is None:
                for metric, key, unit in (("wall_s", "wall_s", "s"), ("output_tokens", "output_tokens", "tokens"),
                                          ("decode_tok_s", "decode_tps", "tok/s"), ("draft_acceptance", "acceptance", "ratio")):
                    if row.get(key) is not None:
                        b.measurements.append(measurement(metric=metric, value=row[key], unit=unit,
                                                          status="measured" if row["eligible"] else "invalid",
                                                          reason=None if row["eligible"] else "weekend summary eligible=false", **common))
            for metric, key, unit in (("round_ms", "round_ms", "ms"), ("tokens_per_round", "tokens_per_round", "tok"),
                                      ("max_depth_tokens", "max_depth", "tokens")):
                if row.get(key) is not None:
                    b.measurements.append(measurement(metric=metric, value=row[key], unit=unit, dims={"arm": row["arm"]},
                                                      status="measured" if row["eligible"] else "invalid",
                                                      reason=None if row["eligible"] else "weekend summary eligible=false", **common))
            b.measurements.append(measurement(metric="weekend_eligible", value=1 if row["eligible"] else 0, unit="bool",
                                              dims={"arm": row["arm"]}, status="measured", **common))
            out.extend(b)
        except Reject as e:
            out.flag(sp, f"row rejected: {e}")
    if not out.measurements:
        raise Reject(f"{sp}: no row could be ingested ({'; '.join(m for _, m in out.flagged[:3])})")
    return out
