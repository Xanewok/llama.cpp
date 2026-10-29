"""The paired comparator, which exists because the unpaired reading gave a wrong answer.

The regression it guards is concrete: a mean over per-task totals reported Q3 at 1.39x Q4
on data whose paired ratio is 1.28x with a sign test of 23/39, i.e. no difference at all.
"""
from __future__ import annotations

import io
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import paired_runs

ROOT = Path(__file__).resolve().parents[1]
SESSION1 = ROOT / "results/2026-08-rental-baseline/codegen/polyglot.jsonl"
SESSION2 = ROOT / "results/2026-08-serving-comparisons/q3-token-cost/polyglot.jsonl"


def row(quant: str, run: int, task: str, tokens: int, ok: bool = True) -> str:
    return json.dumps({"quant": quant, "run_index": run, "task_id": task,
                       "completion_tokens": tokens, "task_success": ok})


def ratios(cand: dict[str, int], base: dict[str, int]) -> list[float]:
    """Per-task cand/base ratios over the tasks both legs hold, as main() forms them."""
    return [cand[t] / base[t] for t in sorted(set(cand) & set(base)) if base[t]]


class LoadTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, name: str, *lines: str) -> Path:
        path = self.dir / name
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_reads_legs_out_of_several_files(self) -> None:
        # The two legs worth comparing were written by different sessions.
        a = self.write("a.jsonl", row("Q4", 1, "t1", 100))
        b = self.write("b.jsonl", row("Q3", 9, "t1", 128))
        self.assertEqual(sorted(paired_runs.load([a, b], 1)), ["t1"])
        self.assertEqual(sorted(paired_runs.load([a, b], 9)), ["t1"])

    def test_quant_disambiguates_a_shared_run_index(self) -> None:
        path = self.write("m.jsonl", row("Q4", 1, "t1", 100), row("IQ3", 1, "t1", 200))
        self.assertEqual(paired_runs.load([path], 1, "IQ3")["t1"]["completion_tokens"], 200)

    def test_an_unnamed_quant_collision_is_refused_not_overwritten(self) -> None:
        # Without this the second quant's rows replace the first's and the leg is a
        # silent mixture of both.
        path = self.write("m.jsonl", row("Q4", 1, "t1", 100), row("IQ3", 1, "t1", 200))
        with self.assertRaises(SystemExit):
            paired_runs.load([path], 1)

    def test_blank_lines_survive(self) -> None:
        path = self.write("b.jsonl", row("A", 1, "t1", 100), "")
        self.assertEqual(len(paired_runs.load([path], 1)), 1)


class SummariseTest(unittest.TestCase):
    def test_an_outlier_does_not_move_the_paired_ratio(self) -> None:
        # The whole point. Both legs cost 2x on every task; one task is 100x larger
        # than the others in absolute terms.
        cand = {f"t{i}": 200 for i in range(9)} | {"t9": 20000}
        base = {f"t{i}": 100 for i in range(9)} | {"t9": 10000}
        self.assertAlmostEqual(paired_runs.summarise(ratios(cand, base))["geometric"], 2.0)

    def test_a_ratio_of_sums_can_disagree_with_every_task(self) -> None:
        # One runaway task makes the ratio of totals say "cand is dearer" when cand was
        # cheaper on 9 tasks of 10. This is the shape of the 1.39x that was reported.
        cand = {f"t{i}": 50 for i in range(9)} | {"t9": 40000}
        base = {f"t{i}": 100 for i in range(9)} | {"t9": 100}
        result = paired_runs.summarise(ratios(cand, base))
        self.assertEqual(result["wins"], 1)
        self.assertLess(result["geometric"], 1.0)
        self.assertGreater(sum(cand.values()) / sum(base.values()), 40)

    def test_a_zero_denominator_is_skipped_not_divided_by(self) -> None:
        cand = {"t1": 1, "t2": 2, "t3": 3}
        base = {"t1": 0, "t2": 2, "t3": 3}
        self.assertEqual(paired_runs.summarise(ratios(cand, base))["n"], 2)


class SignTest(unittest.TestCase):
    def test_a_coin_flip_is_not_significant(self) -> None:
        self.assertGreater(paired_runs.sign_p(23, 39), 0.05)

    def test_a_lopsided_result_is(self) -> None:
        self.assertLess(paired_runs.sign_p(34, 40), 0.005)

    def test_symmetric(self) -> None:
        self.assertAlmostEqual(paired_runs.sign_p(34, 40), paired_runs.sign_p(6, 40))

    def test_bounded_at_one(self) -> None:
        self.assertLessEqual(paired_runs.sign_p(5, 10), 1.0)


class PowerTest(unittest.TestCase):
    def test_smaller_effects_need_more_tasks(self) -> None:
        self.assertLess(paired_runs.tasks_needed(0.62, 1.5), paired_runs.tasks_needed(0.62, 1.15))

    def test_the_measured_spread_reproduces_the_recorded_figure(self) -> None:
        # STATE.md tells the reader 40 tasks resolves 1.3x and nothing finer.
        self.assertLessEqual(paired_runs.tasks_needed(0.622, 1.30), 50)
        self.assertGreater(paired_runs.tasks_needed(0.622, 1.15), 150)


class EndToEndTest(unittest.TestCase):
    def run_cli(self, *argv: str) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            paired_runs.main_argv(list(argv))
        return out.getvalue()

    def test_the_real_results_still_say_q3_is_a_coin_flip(self) -> None:
        text = self.run_cli("--path", str(SESSION1), "--path", str(SESSION2),
                            "--base", "1", "--base-quant", "UD-Q4_K_XL",
                            "--cand", "9", "--cand-quant", "UD-Q3_K_XL", "--solved-only")
        self.assertIn("23/39", text)
        self.assertIn("sign test p", text)
        self.assertIn("paired tasks for 80% power", text)

    def test_iq3_separates_where_q3_does_not(self) -> None:
        text = self.run_cli("--path", str(SESSION1),
                            "--base", "1", "--base-quant", "UD-Q4_K_XL",
                            "--cand", "1", "--cand-quant", "UD-IQ3_XXS", "--solved-only")
        self.assertIn("34/40", text)

    def test_a_leg_holding_two_quants_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_cli("--path", str(SESSION1), "--base", "1", "--cand", "1")


if __name__ == "__main__":
    unittest.main()
