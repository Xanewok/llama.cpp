from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "analyze_paired", ROOT / "scripts" / "analysis" / "analyze_paired.py"
)
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class DiscordantTest(unittest.TestCase):
    def test_counts_each_direction_separately(self) -> None:
        a = {"t1": True, "t2": False, "t3": True, "t4": True}
        b = {"t1": True, "t2": True, "t3": False, "t4": True}
        self.assertEqual(analyze.discordant(a, b), (4, 1, 1))

    def test_ignores_tasks_missing_from_either_side(self) -> None:
        # A crashed task must shrink n, not silently count as agreement.
        a = {"t1": True, "t2": False}
        b = {"t1": False}
        self.assertEqual(analyze.discordant(a, b), (1, 1, 0))


class McNemarTest(unittest.TestCase):
    def test_no_discordant_pairs_is_not_evidence(self) -> None:
        self.assertEqual(analyze.exact_mcnemar_p(0, 0), 1.0)

    def test_symmetric_split_is_not_significant(self) -> None:
        self.assertEqual(analyze.exact_mcnemar_p(5, 5), 1.0)

    def test_lopsided_split_is_significant(self) -> None:
        # 0-vs-8 is the two-sided sign test: 2 * 0.5**8.
        self.assertAlmostEqual(analyze.exact_mcnemar_p(0, 8), 2 * 0.5**8)
        self.assertLess(analyze.exact_mcnemar_p(0, 8), 0.05)

    def test_is_direction_agnostic(self) -> None:
        self.assertEqual(analyze.exact_mcnemar_p(2, 9), analyze.exact_mcnemar_p(9, 2))

    def test_small_n_cannot_reach_significance(self) -> None:
        # Guards the reading error this analysis exists to prevent: at n=4 even a
        # clean sweep is p>0.05, so a 4-task gap is never on its own a result.
        self.assertGreater(analyze.exact_mcnemar_p(0, 4), 0.05)


class TimeoutDerivationTest(unittest.TestCase):
    def test_recovers_the_flag_from_the_note_the_harness_wrote(self) -> None:
        row = analyze.derive_timed_out(
            {"timed_out": None, "notes": "task timed out after 900s"}
        )
        self.assertTrue(row["timed_out"])

    def test_a_test_failure_is_not_a_timeout(self) -> None:
        row = analyze.derive_timed_out(
            {"timed_out": None, "notes": "tests failed after one repair round"}
        )
        self.assertFalse(row["timed_out"])

    def test_an_explicit_flag_is_never_overridden(self) -> None:
        row = analyze.derive_timed_out({"timed_out": False, "notes": "task timed out after 900s"})
        self.assertFalse(row["timed_out"])

    def test_absent_notes_are_not_a_timeout(self) -> None:
        self.assertFalse(analyze.derive_timed_out({"timed_out": None, "notes": None})["timed_out"])


class TimeoutExclusionTest(unittest.TestCase):
    def _rows(self):
        return {
            ("A", 1, "t1"): {"task_success": True, "timed_out": False},
            ("A", 1, "t2"): {"task_success": False, "timed_out": True},
        }

    def test_timed_out_tasks_are_missing_not_failures(self) -> None:
        # Scoring a cut-off as a failure lets a slower quant look less capable.
        self.assertEqual(analyze.outcomes(self._rows(), "A", 1), {"t1": True})

    def test_they_can_be_included_deliberately(self) -> None:
        both = analyze.outcomes(self._rows(), "A", 1, include_timeouts=True)
        self.assertEqual(both, {"t1": True, "t2": False})


class DirectionVsMagnitudeTest(unittest.TestCase):
    """Total disagreement is direction-blind and must not gate a conclusion."""

    def test_symmetric_churn_is_no_evidence_however_large(self) -> None:
        lo, hi = analyze.paired_difference_ci(40, 10, 10)
        self.assertEqual(analyze.exact_mcnemar_p(10, 10), 1.0)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_a_smaller_one_sided_loss_is_evidence(self) -> None:
        # Six one-directional losses: half the disagreement of the case above,
        # yet the only one of the two that supports a conclusion.
        lo, hi = analyze.paired_difference_ci(40, 0, 6)
        self.assertLess(analyze.exact_mcnemar_p(0, 6), 0.05)
        self.assertGreater(lo, 0.0)

    def test_interval_is_symmetric_under_swapping_the_arms(self) -> None:
        a = analyze.paired_difference_ci(40, 3, 9)
        b = analyze.paired_difference_ci(40, 9, 3)
        self.assertAlmostEqual(a[0], -b[1])
        self.assertAlmostEqual(a[1], -b[0])

    def test_no_shared_tasks_yields_no_interval(self) -> None:
        self.assertEqual(analyze.paired_difference_ci(0, 0, 0), (0.0, 0.0))


class LoadTest(unittest.TestCase):
    def test_rows_key_on_quant_run_and_task(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text(
                "\n".join(
                    json.dumps(r)
                    for r in (
                        {"quant": "A", "run_index": 1, "task_id": "t", "task_success": True},
                        {"quant": "A", "run_index": 2, "task_id": "t", "task_success": False},
                        {"quant": "B", "run_index": 1, "task_id": "t", "task_success": True},
                    )
                )
            )
            rows = analyze.load(p)
            self.assertEqual(len(rows), 3, "same task across runs/quants must not collide")
            self.assertEqual(analyze.outcomes(rows, "A", 1), {"t": True})
            self.assertEqual(analyze.outcomes(rows, "A", 2), {"t": False})


if __name__ == "__main__":
    unittest.main()
