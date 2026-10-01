#!/usr/bin/env python3
import collections
import json
import pathlib
import statistics
import sys


def summarize(folder):
    samples = collections.defaultdict(list)
    hashes = collections.defaultdict(set)
    files = sorted(folder.glob("*-d*.jsonl"))
    if len(files) != 12:
        raise ValueError("expected six boots on each of two devices")
    for file in files:
        boot, arm, device = file.stem.split("-")
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        seen = set()
        for row in rows:
            if row["kind"] != "latency":
                continue
            key = (device, row["shape"], row["variant"], row["m"], row["cache"])
            if key in seen:
                raise ValueError(f"duplicate measurement in {file}")
            seen.add(key)
            samples[(arm, key)].append(row["p50_us"])
            hashes[key].add(row["output_sha256"])
        if len(seen) != 288:
            raise ValueError(f"incomplete shape matrix in {file}: {len(seen)}")
    if any(len(h) != 1 for h in hashes.values()):
        raise ValueError("up/patch output differs; exactness gate failed")
    result = []
    for key in sorted(hashes):
        up, patch = samples[("up", key)], samples[("patch", key)]
        if len(up) != 3 or len(patch) != 3:
            raise ValueError("missing arm boots")
        result.append(dict(key=key, up_us_M=statistics.median(up), patch_us_M=statistics.median(patch),
                           delta_us_D=statistics.median(patch) - statistics.median(up)))
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(pathlib.Path(sys.argv[1])), indent=2))
