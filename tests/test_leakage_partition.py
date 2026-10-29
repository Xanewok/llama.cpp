from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.corpora import make_subsets


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _l1_repos() -> set[str]:
    sources = json.loads((ROOT / "corpora" / "sources.json").read_text())
    return {
        item["repo"].casefold()
        for language in ("rust", "js-ts")
        for item in sources[language]
    }


def _l2_repos() -> set[str]:
    # Polyglot contributes only its own container repo, which cannot collide with
    # anything; asserting on it would pass vacuously. Diff-XYZ rows carry the real
    # upstream repository each diff came from, so that is the L2 exposure.
    rows = _jsonl(ROOT / "subsets" / "diffxyz.jsonl")
    return {row["repo"].casefold() for row in rows if row.get("repo")}


def _l3_repos() -> set[str]:
    return {row["repo"].casefold() for row in _jsonl(ROOT / "subsets" / "swebench.jsonl")}


class LeakagePartitionTest(unittest.TestCase):
    def test_layers_draw_from_disjoint_repositories(self) -> None:
        l1, l2, l3 = _l1_repos(), _l2_repos(), _l3_repos()

        # An empty or singleton set is disjoint from everything, so a selection bug
        # that emptied a layer would look like an enforced partition.
        self.assertGreaterEqual(len(l1), 10, "L1 repo set implausibly small")
        self.assertGreaterEqual(len(l2), 20, "L2 upstream repo set implausibly small")
        self.assertGreaterEqual(len(l3), 3, "L3 repo set implausibly small")

        for name, overlap in (("L1/L2", l1 & l2), ("L1/L3", l1 & l3), ("L2/L3", l2 & l3)):
            self.assertFalse(overlap, f"{name} share repositories: {sorted(overlap)}")

    def test_overlap_detection_actually_fires(self) -> None:
        # The assertion above must be capable of failing, or it enforces nothing.
        l1 = _l1_repos()
        colliding = {next(iter(sorted(l1)))}
        self.assertTrue(l1 & colliding, "overlap detection failed on a planted collision")

    def test_agentic_subset_is_fifty_two_distinct_instances(self) -> None:
        # 50 is the floor a paired McNemar design needs to separate a 10 pp pass-rate
        # gap; unpaired the same gap would take ~376 per arm.
        rows = _jsonl(ROOT / "subsets" / "swebench.jsonl")
        self.assertEqual(len(rows), 52)
        self.assertEqual(len({row["instance_id"] for row in rows}), 52)

    def test_the_agentic_subset_still_contains_the_original_ten(self) -> None:
        # Widening must extend the instance list, never resample it: legs already run
        # against the first ten stay comparable only if those ten are still in it.
        rows = _jsonl(ROOT / "subsets" / "swebench.jsonl")
        self.assertTrue(
            set(make_subsets.ORIGINAL_AGENTIC_IDS) <= {row["instance_id"] for row in rows}
        )

    def test_no_repo_supplies_more_than_three_instances(self) -> None:
        rows = _jsonl(ROOT / "subsets" / "swebench.jsonl")
        worst = Counter(row["repo"] for row in rows).most_common(1)[0]
        self.assertLessEqual(worst[1], 3, worst)

    def test_every_instance_stays_inside_the_discriminating_window(self) -> None:
        # Outside 0.25-0.80 an instance scores the same for every quant and separates
        # nothing, which is the whole reason the subset is filtered rather than sampled.
        for row in _jsonl(ROOT / "subsets" / "swebench.jsonl"):
            self.assertGreaterEqual(row["solve_rate"], 0.25, row["instance_id"])
            self.assertLessEqual(row["solve_rate"], 0.80, row["instance_id"])


if __name__ == "__main__":
    unittest.main()
