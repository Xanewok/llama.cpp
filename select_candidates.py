"""Deterministic candidate selection for the swe-fresh pool (fable-bench REPORT 2.2, step 1).

CPU only. Inputs: the HF parquet of one SWE-rebench-leaderboard split and Docker Hub tag sizes.
Output: a JSON that freezes ids, images, digests and the filter, with the input hash.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path
import urllib.request

import pyarrow.parquet as pq

PARQUET_URL = "https://huggingface.co/api/datasets/nebius/SWE-rebench-leaderboard/parquet/default/{split}/0.parquet"
HUB_TAG = "https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"
GB = 10**9


def rank(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def patch_shape(patch):
    return patch.count("\n"), patch.count("diff --git ")


def eligible(row, size, max_patch_lines, max_files, max_statement, max_image_gb, max_memory_gb):
    lines, files = patch_shape(row["patch"])
    memory = row.get("harbor_memory")
    reasons = []
    if lines > max_patch_lines:
        reasons.append("patch_lines")
    if files > max_files:
        reasons.append("patch_files")
    if len(row["problem_statement"]) > max_statement:
        reasons.append("statement_chars")
    if size is None or size > max_image_gb * GB:
        reasons.append("image_size")
    if isinstance(memory, (int, float)) and memory > max_memory_gb * 1024:
        reasons.append("harbor_memory")
    return reasons


def pick(rows, sizes, seed, count, **limits):
    """One instance per repo (lowest seeded hash), then the `count` lowest hashes overall."""
    ok = [r for r in rows if not eligible(r, sizes.get(r["instance_id"], {}).get("size"), **limits)]
    best = {}
    for row in ok:
        key = rank(seed, row["instance_id"])
        if row["repo"] not in best or key < best[row["repo"]][0]:
            best[row["repo"]] = (key, row)
    chosen = sorted(best.values(), key=lambda pair: pair[0])[:count]
    return [row for _, row in chosen], len(ok), len(best)


def hub_sizes(rows, workers=8):
    def one(row):
        repo, tag = row["docker_image"].split(":")
        try:
            data = json.load(urllib.request.urlopen(HUB_TAG.format(repo=repo, tag=tag), timeout=30))
            return row["instance_id"], {"size": data["full_size"], "digest": data.get("digest")}
        except Exception as error:  # a missing tag must exclude the row, not abort the selection
            return row["instance_id"], {"size": None, "digest": None, "error": str(error)}

    with futures.ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--sizes", type=Path, help="cache of Docker Hub sizes; written when absent")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-patch-lines", type=int, default=150)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--max-statement", type=int, default=6000)
    parser.add_argument("--max-image-gb", type=float, default=2.5)
    parser.add_argument("--max-memory-gb", type=float, default=4.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = pq.read_table(args.parquet).to_pylist()
    if args.sizes and args.sizes.exists():
        sizes = json.loads(args.sizes.read_text())
    else:
        sizes = hub_sizes(rows)
        if args.sizes:
            args.sizes.write_text(json.dumps(sizes, indent=1))
    limits = dict(max_patch_lines=args.max_patch_lines, max_files=args.max_files,
                  max_statement=args.max_statement, max_image_gb=args.max_image_gb,
                  max_memory_gb=args.max_memory_gb)
    chosen, eligible_count, repos = pick(rows, sizes, args.seed, args.count, **limits)
    document = {
        "split_rows": len(rows), "eligible": eligible_count, "eligible_repos": repos, "seed": args.seed,
        "filter": limits, "parquet_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
        "tasks": [{"instance_id": r["instance_id"], "repo": r["repo"], "created_at": r["created_at"],
                   "docker_image": r["docker_image"], "image_gb": sizes[r["instance_id"]]["size"] / GB,
                   "image_digest": sizes[r["instance_id"]]["digest"],
                   "patch_lines": patch_shape(r["patch"])[0], "patch_files": patch_shape(r["patch"])[1],
                   "statement_chars": len(r["problem_statement"]),
                   "interface_variant": r["instance_id"].endswith("_interface")} for r in chosen]}
    args.out.write_text(json.dumps(document, indent=1) + "\n")
    print(json.dumps({k: v for k, v in document.items() if k != "tasks"}))
    print("\n".join(t["instance_id"] for t in document["tasks"]))


if __name__ == "__main__":
    main()
