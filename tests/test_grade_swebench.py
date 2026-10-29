"""Grading maps the harness's verdict onto rows, and refuses where it has none.

The failure this guards is the one the repo keeps paying for: an infra failure and a
wrong patch are the same empty result to anything that counts successes, and only one
of them belongs to the quant under test.
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

from scripts.analysis import grade_swebench


def report(**kwargs) -> dict:
    base = {k: [] for k in ("resolved_ids", "unresolved_ids", "empty_patch_ids",
                            "error_ids", "incomplete_ids", "infra_failure_ids",
                            "ambiguous_failure_ids")}
    return base | kwargs


def row(task_id: str, **extra) -> dict:
    return {"task_id": task_id, "quant": "Q4", "run_index": 1,
            "task_success": None, "tests_passed": None, "notes": None} | extra


class VerdictTest(unittest.TestCase):
    def test_resolved_is_a_solve(self) -> None:
        self.assertEqual(grade_swebench.verdicts(report(resolved_ids=["a"]))["a"], (True, None))

    def test_unresolved_is_a_failure(self) -> None:
        self.assertEqual(grade_swebench.verdicts(report(unresolved_ids=["a"]))["a"], (False, None))

    def test_an_empty_patch_is_the_models_failure(self) -> None:
        # The model was asked and produced nothing. That is a result, not a gap.
        self.assertEqual(grade_swebench.verdicts(report(empty_patch_ids=["a"]))["a"], (False, None))

    def test_an_infra_failure_is_not_charged_to_the_model(self) -> None:
        # The harness leaves infra failures in unresolved_ids as well; reading that
        # list first would score a container that never built as a wrong patch.
        table = grade_swebench.verdicts(
            report(unresolved_ids=["a"], infra_failure_ids=["a"])
        )
        self.assertEqual(table["a"], (None, "infra_failure"))

    def test_an_ambiguous_failure_stays_an_unresolved_instance(self) -> None:
        # The harness separates ambiguous from confirmed-environment faults so the
        # ambiguous ones are never read as confirmed, and keeps them in unresolved_ids
        # on purpose. Excluding them here would discard failures the patch may have
        # caused -- the opposite bias to charging infra failures to the model.
        table = grade_swebench.verdicts(report(unresolved_ids=["a"], ambiguous_failure_ids=["a"]))
        self.assertEqual(table["a"], (False, None))

    def test_an_error_has_no_verdict(self) -> None:
        self.assertEqual(grade_swebench.verdicts(report(error_ids=["a"]))["a"], (None, "error"))

    def test_resolved_beats_a_stray_classification(self) -> None:
        # Tests that pass are the definition of solved, whatever else was logged.
        table = grade_swebench.verdicts(report(resolved_ids=["a"], ambiguous_failure_ids=["a"]))
        self.assertEqual(table["a"], (True, None))


class GradeTest(unittest.TestCase):
    def test_verdicts_reach_both_fields(self) -> None:
        graded, tally = grade_swebench.grade([row("a")], report(resolved_ids=["a"]))
        self.assertTrue(graded[0]["task_success"])
        self.assertTrue(graded[0]["tests_passed"])
        self.assertEqual(tally["resolved"], 1)

    def test_a_row_the_report_never_mentions_stays_ungraded(self) -> None:
        graded, tally = grade_swebench.grade([row("a")], report())
        self.assertIsNone(graded[0]["task_success"])
        self.assertIn("absent from the report", graded[0]["notes"])
        self.assertEqual(tally["ungraded"], 1)

    def test_an_existing_note_is_kept(self) -> None:
        graded, _ = grade_swebench.grade([row("a", notes="Codex produced an empty diff")],
                                         report(error_ids=["a"]))
        self.assertIn("Codex produced an empty diff", graded[0]["notes"])
        self.assertIn("ungraded: error", graded[0]["notes"])

    def test_the_input_rows_are_not_mutated(self) -> None:
        # --rows is the append-only record of what the run measured.
        original = row("a")
        grade_swebench.grade([original], report(resolved_ids=["a"]))
        self.assertIsNone(original["task_success"])


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.rows = self.dir / "leg.jsonl"
        self.rows.write_text("\n".join(json.dumps(r) for r in [
            row("a"), row("b"), row("c"),
            row("a", quant="IQ3"),  # a different leg in the same file
        ]) + "\n")
        self.report = self.dir / "r.json"
        self.report.write_text(json.dumps(report(
            resolved_ids=["a"], unresolved_ids=["b", "c"], infra_failure_ids=["c"])))

    def run_cli(self, *argv: str) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            grade_swebench.main(list(argv))
        return out.getvalue()

    def base(self) -> list[str]:
        return ["--rows", str(self.rows), "--quant", "Q4", "--run-index", "1",
                "--report", str(self.report), "--out", str(self.dir / "g.jsonl")]

    def test_grades_only_the_named_leg(self) -> None:
        self.run_cli(*self.base())
        graded = [json.loads(l) for l in (self.dir / "g.jsonl").read_text().splitlines()]
        self.assertEqual({r["quant"] for r in graded}, {"Q4"})
        self.assertEqual(len(graded), 3)

    def test_the_pass_rate_excludes_ungraded_instances(self) -> None:
        # 1 resolved, 1 unresolved, 1 infra failure -> 1/2, not 1/3.
        text = self.run_cli(*self.base())
        self.assertIn("pass rate 0.500 over 2 graded", text)
        self.assertIn("1 ungraded", text)

    def test_an_unknown_leg_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_cli("--rows", str(self.rows), "--quant", "Q6", "--run-index", "1",
                         "--report", str(self.report))

    def test_report_and_run_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_cli(*self.base(), "--run")

    def test_a_file_that_is_not_a_report_is_refused(self) -> None:
        bad = self.dir / "bad.json"
        bad.write_text(json.dumps({"hello": "world"}))
        with self.assertRaises(SystemExit):
            self.run_cli("--rows", str(self.rows), "--quant", "Q4", "--run-index", "1",
                         "--report", str(bad))


class InfraFailureTest(unittest.TestCase):
    """An instance that never reached the model must not be graded, resumed past, or
    submitted as an empty patch -- three separate ways one bad clone becomes a data point
    attributed to the quant under test."""

    def setUp(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.instruments import run_swebench
        self.run_swebench = run_swebench
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_an_infra_failure_is_marked_off_the_row(self) -> None:
        result = self.run_swebench.InstanceResult("x", "", {}, infra_failure=True)
        self.assertNotIn("infra_failure", result.measures)

    def test_resume_retries_an_infra_failure_but_not_an_empty_diff(self) -> None:
        out = self.dir / "rows.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in [
            {"task_id": "clone-died", "quant": "Q", "run_index": 1,
             "task_success": None, "notes": "FileNotFoundError: disk full"},
            {"task_id": "no-diff", "quant": "Q", "run_index": 1,
             "task_success": None, "notes": "Codex produced an empty diff"},
            {"task_id": "other-leg", "quant": "IQ3", "run_index": 1,
             "task_success": None, "notes": "OSError: boom"},
        ]) + "\n")
        self.assertEqual(self.run_swebench.infra_failed_ids(out, "Q", 1), {"clone-died"})


class HarnessCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_the_run_is_scoped_to_the_submitted_instances(self) -> None:
        # The pinned export holds 300 instances; without --instance_ids the harness
        # reports the 248 nobody submitted as incomplete, hiding a real omission.
        predictions = self.dir / "p.jsonl"
        predictions.write_text("\n".join(json.dumps(
            {"instance_id": i, "model_name_or_path": "m", "model_patch": "x"})
            for i in ("b", "a")) + "\n")
        seen = {}

        def fake_run(command, check):
            seen["command"] = command
            (self.dir / "reports").mkdir(exist_ok=True)
            (self.dir / "reports" / "m.rid.json").write_text(json.dumps(report()))
            return None

        original = grade_swebench.subprocess.run
        grade_swebench.subprocess.run = fake_run
        self.addCleanup(lambda: setattr(grade_swebench.subprocess, "run", original))
        args = grade_swebench.build_parser().parse_args([
            "--rows", str(predictions), "--quant", "q", "--run-index", "0",
            "--predictions", str(predictions), "--run",
            "--report-dir", str(self.dir / "reports"), "--run-id", "rid"])
        grade_swebench.run_harness(args)
        command = seen["command"]
        at = command.index("--instance_ids")
        self.assertEqual(command[at + 1:at + 3], ["a", "b"])

    def test_an_empty_prediction_file_is_refused(self) -> None:
        predictions = self.dir / "p.jsonl"
        predictions.write_text("")
        args = grade_swebench.build_parser().parse_args([
            "--rows", str(predictions), "--quant", "q", "--run-index", "0",
            "--predictions", str(predictions), "--run", "--run-id", "rid"])
        with self.assertRaises(SystemExit):
            grade_swebench.run_harness(args)


if __name__ == "__main__":
    unittest.main()
