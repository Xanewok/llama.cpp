"""Recompute CPU timing summaries from retained raw events; no inference or GPU access."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
for fixture in ("synthetic", "real"):
    directory = root / "reuse-results" / fixture
    records = json.loads((directory / "results.json").read_text())
    for record in records:
        label = f"m{record['meta']}-k{record['k']}-s{record['seed']}-{record['boot']}-{record['arm']}"
        rows = [json.loads(line) for line in (directory / (label + ".jsonl")).read_text().splitlines()][3:]
        record["decode_wall_us_per_round_D"] = statistics.mean(sum(p["end_ns"] - p["begin_ns"] for p in row["phases"]) / 1000 for row in rows)
    summary = json.loads((root / (fixture + "-reuse-results.json")).read_text())
    for item in summary["summary"]:
        selected = [r for r in records if (r["meta"], r["k"], r["arm"]) == (item["meta"], item["k"], item["arm"])]
        item["decode_wall_us_per_round_median_D"] = statistics.median(r["decode_wall_us_per_round_D"] for r in selected)
    (root / (fixture + "-reuse-results.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(fixture)
    for item in summary["summary"]:
        print(item["meta"], item["k"], item["arm"], round(item["C_D1_build_union_us_per_round_median_D"], 1), round(item["decode_wall_us_per_round_median_D"], 1))
