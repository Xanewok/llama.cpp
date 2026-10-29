from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.instruments import bench_mtp  # noqa: E402


def _tasks() -> list[dict]:
    return sorted(
        json.loads((ROOT / "subsets" / "polyglot.json").read_text()),
        key=lambda t: t["task_id"],
    )


class AcceptanceTest(unittest.TestCase):
    def test_no_drafts_yields_no_rate(self) -> None:
        # An MTP-off row must stay empty rather than report a rate of zero, which
        # would read as "the head drafted and was always wrong".
        self.assertEqual(bench_mtp.acceptance({"drafts": 0, "draft_tokens": 0}), {})

    def test_rates_are_per_token_and_per_draft(self) -> None:
        got = bench_mtp.acceptance(
            {"drafts": 100, "draft_tokens": 300, "draft_accepted_tokens": 180}
        )
        self.assertAlmostEqual(got["draft_accept_rate"], 0.6)
        self.assertAlmostEqual(got["accepted_per_draft"], 1.8)


class BenchPromptTest(unittest.TestCase):
    def test_prompt_carries_stub_and_tests(self) -> None:
        for task in _tasks()[:2] + [t for t in _tasks() if t["language"] == "rust"][:2]:
            prompt = bench_mtp.bench_prompt(task, ROOT / "subsets")
            self.assertIn(task["solution_path"][0], prompt, task["task_id"])
            self.assertRegex(prompt, r"(?i)(test|spec)", task["task_id"])

    def test_reference_solution_never_reaches_the_prompt(self) -> None:
        # .meta holds the worked answer. Leaking it would measure copying, not
        # generation, and draft acceptance on copied text is meaninglessly high.
        for task in _tasks():
            base = (ROOT / "subsets" / task["source_dir"]).resolve()
            reference = base / task["example_path"][0]
            if not reference.is_file():
                continue
            body = reference.read_text(errors="replace").strip()
            self.assertNotIn(body, bench_mtp.bench_prompt(task, ROOT / "subsets"), task["task_id"])

    def test_the_meta_exclusion_is_not_vacuous(self) -> None:
        # If no task shipped a .meta reference, the guard above would pass on nothing.
        found = [
            t
            for t in _tasks()
            if ((ROOT / "subsets" / t["source_dir"]).resolve() / t["example_path"][0]).is_file()
        ]
        self.assertGreaterEqual(len(found), 20, "no .meta references present to exclude")


if __name__ == "__main__":
    unittest.main()
